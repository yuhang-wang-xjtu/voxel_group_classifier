"""
Sonata v1m1 Base

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from itertools import chain
from packaging import version
from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch_scatter
from timm.layers import trunc_normal_

import pointops
from pointcept.models.utils.structure import Point
from pointcept.models.builder import MODELS, build_model
from pointcept.models.modules import PointModel
from pointcept.models.utils import offset2batch, offset2bincount, batch2offset
from pointcept.utils.comm import get_world_size, all_gather
from pointcept.utils.scheduler import CosineScheduler
from pointcept.models.sonata.voxel_group_classifier import (
    VoxelGroupClassifier,
    MaskDiversityScheduler,
)


class OnlineCluster(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=4096,
        embed_channels=512,
        num_prototypes=4096,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, embed_channels),
        )
        self.apply(self._init_weights)
        if version.parse(torch.__version__) >= version.parse("2.1.0"):
            self.prototype = torch.nn.utils.parametrizations.weight_norm(
                nn.Linear(embed_channels, num_prototypes, bias=False)
            )
            self.prototype.parametrizations.weight.original0.data.fill_(1)
            self.prototype.parametrizations.weight.original0.requires_grad = False

        else:
            self.prototype = torch.nn.utils.weight_norm(
                nn.Linear(embed_channels, num_prototypes, bias=False)
            )
            self.prototype.weight_g.data.fill_(1)
            self.prototype.weight_g.requires_grad = False

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, feat):
        feat = self.mlp(feat)
        eps = 1e-6 if feat.dtype == torch.float16 else 1e-12
        feat = nn.functional.normalize(feat, dim=-1, p=2, eps=eps)
        similarity = self.prototype(feat)
        return similarity


@MODELS.register_module("Sonata-v1m2-vgc")
class SonataVGC(PointModel):
    def __init__(
        self,
        backbone,
        head_in_channels,
        head_hidden_channels=4096,
        head_embed_channels=512,
        head_num_prototypes=4096,
        teacher_custom=None,
        num_global_view=2,
        num_local_view=4,
        mask_size_start=0.1,
        mask_size_base=0.4,
        mask_size_warmup_ratio=0.05,
        mask_ratio_start=0.3,
        mask_ratio_base=0.7,
        mask_ratio_warmup_ratio=0.05,
        mask_jitter=None,
        teacher_temp_start=0.04,
        teacher_temp_base=0.07,
        teacher_temp_warmup_ratio=0.05,
        student_temp=0.1,
        mask_loss_weight=2 / 8,
        roll_mask_loss_weight=2 / 8,
        unmask_loss_weight=4 / 8,
        momentum_base=0.996,
        momentum_final=1,
        match_max_k=8,
        match_max_r=0.08,
        up_cast_level=2,
        masking_mode="random",
        vgc_embed_dim=512,
        vgc_hidden_dim=128,
        vgc_num_groups=None,
        vgc_label_smooth=0.1,
        vgc_temperature=1.0,
        vgc_target_temp=2.0,
        vgc_loss_weight=0.1,
        vgc_warmup_ratio=0.2,
        vgc_diversity_decay=0.9,
        vgc_diversity_max_penalty=0.5,
    ):
        super(Sonata, self).__init__()
        self.masking_mode = masking_mode
        self.mask_loss_weight = mask_loss_weight
        self.roll_mask_loss_weight = roll_mask_loss_weight
        self.unmask_loss_weight = unmask_loss_weight

        self.num_global_view = num_global_view
        self.num_local_view = num_local_view

        # masking and scheduler
        self.mask_size = mask_size_start
        self.mask_size_start = mask_size_start
        self.mask_size_base = mask_size_base
        self.mask_size_warmup_ratio = mask_size_warmup_ratio
        self.mask_size_scheduler = None

        self.mask_ratio = mask_ratio_start
        self.mask_ratio_start = mask_ratio_start
        self.mask_ratio_base = mask_ratio_base
        self.mask_ratio_warmup_ratio = mask_ratio_warmup_ratio
        self.mask_ratio_scheduler = None

        self.mask_jitter = mask_jitter

        # temperature and scheduler
        self.teacher_temp = teacher_temp_start
        self.teacher_temp_start = teacher_temp_start
        self.teacher_temp_base = teacher_temp_base
        self.teacher_temp_warmup_ratio = teacher_temp_warmup_ratio
        self.teacher_temp_scheduler = None
        self.student_temp = student_temp

        # momentum and scheduler
        self.momentum = momentum_base
        self.momentum_base = momentum_base
        self.momentum_final = momentum_final
        self.momentum_scheduler = None

        # dynamic matching
        self.match_max_k = match_max_k
        self.match_max_r = match_max_r

        # up cast level
        self.up_cast_level = up_cast_level

        # one of unmask, mask, roll mask loss enable
        assert unmask_loss_weight + mask_loss_weight + roll_mask_loss_weight > 0
        # roll mask loss need more than one global view
        assert num_global_view > 1 or roll_mask_loss_weight == 0
        # current roll mask only support two global views
        assert num_global_view == 1 or num_global_view == 2

        student_model_dict = dict()
        teacher_model_dict = dict()
        if teacher_custom is None:
            teacher_custom = {}
        student_backbone = build_model(backbone)
        # turn off parameters like drop path for teacher model
        backbone.update(teacher_custom)

        teacher_backbone = build_model(backbone)
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone

        head = partial(
            OnlineCluster,
            in_channels=head_in_channels,
            hidden_channels=head_hidden_channels,
            embed_channels=head_embed_channels,
            num_prototypes=head_num_prototypes,
        )
        if self.mask_loss_weight > 0 or self.roll_mask_loss_weight > 0:
            student_model_dict["mask_head"] = head()
            # teacher share only one head and EMA update by student.mask_head (global)
            teacher_model_dict["mask_head"] = head()

        if self.unmask_loss_weight > 0:
            student_model_dict["unmask_head"] = head()
            # dummy head for EMA update implementation
            teacher_model_dict["unmask_head"] = head()

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)
        for k, v in self.student.items():
            self.teacher[k].load_state_dict(self.student[k].state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False

        # VoxelGroupClassifier for entropy-guided adaptive masking
        self.vgc = None
        self.vgc_diversity = None
        self.vgc_warmup_ratio = vgc_warmup_ratio
        self.vgc_loss_weight = vgc_loss_weight
        self.vgc_target_temp = vgc_target_temp
        if masking_mode == "vgc":
            num_groups = vgc_num_groups or head_num_prototypes
            self.vgc = VoxelGroupClassifier(
                embed_dim=vgc_embed_dim,
                hidden_dim=vgc_hidden_dim,
                num_groups=num_groups,
                label_smooth=vgc_label_smooth,
                temperature=vgc_temperature,
                loss_weight=1.0,
            )
            self.vgc_diversity = MaskDiversityScheduler(
                decay=vgc_diversity_decay,
                max_penalty=vgc_diversity_max_penalty,
            )
        self._vgc_alpha = 0.0

    def before_train(self):
        # make ModelHook after CheckPointLoader
        total_steps = self.trainer.cfg.scheduler.total_steps
        curr_step = self.trainer.start_epoch * len(self.trainer.train_loader)
        # mask size scheduler
        self.mask_size_scheduler = CosineScheduler(
            start_value=self.mask_size_start,
            base_value=self.mask_size_base,
            final_value=self.mask_size_base,
            warmup_iters=int(total_steps * self.mask_size_warmup_ratio),
            total_iters=total_steps,
        )
        self.mask_size_scheduler.iter = curr_step

        # mask ratio scheduler
        self.mask_ratio_scheduler = CosineScheduler(
            start_value=self.mask_ratio_start,
            base_value=self.mask_ratio_base,
            final_value=self.mask_ratio_base,
            warmup_iters=int(total_steps * self.mask_ratio_warmup_ratio),
            total_iters=total_steps,
        )
        self.mask_ratio_scheduler.iter = curr_step

        # teacher temperature scheduler
        self.teacher_temp_scheduler = CosineScheduler(
            start_value=self.teacher_temp_start,
            base_value=self.teacher_temp_base,
            final_value=self.teacher_temp_base,
            warmup_iters=int(total_steps * self.teacher_temp_warmup_ratio),
            total_iters=total_steps,
        )
        self.teacher_temp_scheduler.iter = curr_step

        # momentum scheduler
        self.momentum_scheduler = CosineScheduler(
            base_value=self.momentum_base,
            final_value=self.momentum_final,
            total_iters=total_steps,
        )
        self.momentum_scheduler.iter = curr_step

    def before_step(self):
        # update parameters from schedulers
        self.mask_size = self.mask_size_scheduler.step()
        self.mask_ratio = self.mask_ratio_scheduler.step()
        self.teacher_temp = self.teacher_temp_scheduler.step()
        self.momentum = self.momentum_scheduler.step()

        # VGC curriculum alpha
        if self.vgc is not None:
            warmup_epochs = max(1, int(
                self.trainer.cfg.epoch * self.vgc_warmup_ratio
            ))
            current_epoch = getattr(self.trainer, 'epoch', 0)
            if current_epoch < warmup_epochs:
                self._vgc_alpha = 0.0
            else:
                progress = (current_epoch - warmup_epochs) / max(1, warmup_epochs)
                self._vgc_alpha = 0.5 * (1.0 - math.cos(min(progress, 1.0) * math.pi))
        else:
            self._vgc_alpha = 0.0

        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar(
                "params/mask_size",
                self.mask_size,
                self.mask_size_scheduler.iter,
            )
            self.trainer.writer.add_scalar(
                "params/mask_ratio",
                self.mask_ratio,
                self.mask_ratio_scheduler.iter,
            )
            self.trainer.writer.add_scalar(
                "params/teacher_temp",
                self.teacher_temp,
                self.teacher_temp_scheduler.iter,
            )
            self.trainer.writer.add_scalar(
                "params/momentum",
                self.momentum,
                self.momentum_scheduler.iter,
            )

    def after_step(self):
        # EMA update teacher
        with torch.no_grad():
            m = self.momentum
            student_param_list = list(self.student.parameters())
            teacher_param_list = list(self.teacher.parameters())
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    @staticmethod
    def sinkhorn_knopp(feat, temp, num_iter=3):
        feat = feat.float()
        q = torch.exp(feat / temp).t()
        n = sum(all_gather(q.shape[1]))  # number of samples to assign
        k = q.shape[0]  # number of prototypes

        # make the matrix sums to 1
        sum_q = q.sum()
        if get_world_size() > 1:
            dist.all_reduce(sum_q)
        q = q / sum_q

        for i in range(num_iter):
            # normalize each row: total weight per prototype must be 1/k
            q_row_sum = q.sum(dim=1, keepdim=True)
            if get_world_size() > 1:
                dist.all_reduce(q_row_sum)
            q = q / q_row_sum / k

            # normalize each column: total weight per sample must be 1/n
            q = q / q.sum(dim=0, keepdim=True) / n

        q *= n  # the columns must sum to 1 so that Q is an assignment
        return q.t()

    def generate_mask(self, coord, offset):
        batch = offset2batch(offset)
        mask_size = self.mask_size
        mask_ratio = self.mask_ratio

        min_coord = torch_scatter.segment_coo(coord, batch, reduce="min")
        grid_coord = ((coord - min_coord[batch]) // mask_size).int()
        grid_coord = torch.cat([batch.unsqueeze(-1), grid_coord], dim=-1)
        unique, point_cluster, counts = torch.unique(
            grid_coord, dim=0, sorted=True, return_inverse=True, return_counts=True
        )
        patch_num = unique.shape[0]
        mask_patch_num = int(patch_num * mask_ratio)

        if self.masking_mode == "vgc" and self.vgc is not None:
            return self._generate_mask_vgc(
                coord, offset, grid_coord, point_cluster, patch_num,
                mask_patch_num, mask_ratio, mask_size,
            )
        elif self.masking_mode == "cosine":
            return self._generate_mask_cosine(
                coord, offset, grid_coord, point_cluster, patch_num,
                mask_patch_num, mask_ratio, mask_size,
            )
        else:
            patch_index = torch.randperm(patch_num, device=coord.device)
            mask_patch_index = patch_index[:mask_patch_num]
            point_mask = torch.isin(point_cluster, mask_patch_index)
        return point_mask, point_cluster

    def _generate_mask_cosine(
        self, coord, offset, grid_coord, point_cluster, patch_num,
        mask_patch_num, mask_ratio, mask_size,
    ):
        """
        Mask selection based on teacher-student cosine similarity.

        Uses cos(t_i, s_i) computed in the PREVIOUS forward pass:
            t_i = teacher prototype logits for token i
            s_i = student prediction logits for token i
            difficulty_i = 1 - cos(t_i, s_i)

        High difficulty = student can't predict from masked context
                       = this region needs more learning → mask.

        On the very first forward, difficulty is unknown → random mask.
        """
        if self._cosine_patch_difficulty is None:
            # First forward: no history → random mask
            patch_index = torch.randperm(patch_num, device=coord.device)
            mask_patch_index = patch_index[:mask_patch_num]
            point_mask = torch.isin(point_cluster, mask_patch_index)
            self._cosine_point_cluster = point_cluster
            return point_mask, point_cluster

        # Use stored difficulty from previous forward
        alpha = getattr(self, "_vgc_alpha", 0.5)
        patch_difficulty = self._cosine_patch_difficulty

        # Add small noise for tie-breaking
        noise = torch.rand(patch_num, device=coord.device) * 1e-6

        if alpha < 0.1:
            sorted_idx = torch.randperm(patch_num, device=coord.device)
        else:
            random_ranking = torch.rand(patch_num, device=coord.device)
            difficulty_ranking = patch_difficulty + noise
            blended = (1.0 - alpha) * random_ranking + alpha * difficulty_ranking
            _, sorted_idx = torch.sort(blended, descending=True)

        mask_patch_index = sorted_idx[:mask_patch_num]
        point_mask = torch.isin(point_cluster, mask_patch_index)

        # Store point_cluster for next forward's aggregation
        self._cosine_point_cluster = point_cluster

        return point_mask, point_cluster

    def _generate_mask_vgc(
        self, coord, offset, grid_coord, point_cluster, patch_num,
        mask_patch_num, mask_ratio, mask_size,
    ):
        """
        Entropy-guided mask generation using VoxelGroupClassifier.

        Runs a lightweight teacher forward to obtain features for
        difficulty estimation. Uses VGC prediction entropy to rank
        grid patches by learning difficulty, then selects the most
        difficult patches for masking.

        To prevent repeatedly masking the same region, a diversity
        penalty is applied based on recent mask history.
        """
        batch = offset2batch(offset)

        # --- Quick teacher forward for feature extraction ---
        with torch.no_grad():
            point = Point(
                coord=coord,
                offset=offset,
                grid_size=mask_size,
            )
            teacher_feat = self.teacher.backbone(point)
            teacher_feat = self.up_cast(teacher_feat)

        # --- Per-point entropy via VGC ---
        features = teacher_feat.feat  # [N, D]
        difficulty_per_point = self.vgc.get_difficulty(
            features.unsqueeze(1),  # [N, 1, D] treat each point as 1 subvoxel
            topk_percentile=1.0,     # only 1 subvoxel per point
        )  # [N]

        # --- Aggregate to per-grid-patch difficulty ---
        patch_difficulty = torch.zeros(patch_num, device=coord.device)
        patch_difficulty.scatter_add_(
            0, point_cluster, difficulty_per_point
        )
        patch_point_count = torch.zeros(patch_num, device=coord.device)
        patch_point_count.scatter_add_(
            0, point_cluster, torch.ones_like(difficulty_per_point)
        )
        patch_difficulty = patch_difficulty / patch_point_count.clamp(min=1)

        # --- Diversity penalty (prevent repeated masking) ---
        if self.vgc_diversity is not None:
            # Create mask-like tensor for current patches (0 = not masked yet)
            # Diversity scheduler tracks which regions were recently masked
            penalty = self.vgc_diversity.penalty()
            if penalty is not None and penalty.shape[0] == patch_num:
                patch_difficulty = patch_difficulty * (1.0 - penalty)

        # --- Select hardest patches to mask ---
        # Add small noise for tie-breaking
        noise = torch.rand(patch_num, device=coord.device) * 1e-6

        alpha = getattr(self, "_vgc_alpha", 1.0)
        if alpha < 0.05:
            # Pure random masking during warmup
            sorted_idx = torch.randperm(patch_num, device=coord.device)
        else:
            # Blend random and difficulty-based ranking
            random_ranking = torch.rand(patch_num, device=coord.device)
            difficulty_ranking = patch_difficulty + noise
            blended = (1.0 - alpha) * random_ranking + alpha * difficulty_ranking
            _, sorted_idx = torch.sort(blended, descending=True)

        mask_patch_index = sorted_idx[:mask_patch_num]
        point_mask = torch.isin(point_cluster, mask_patch_index)

        # --- Update diversity history ---
        if self.vgc_diversity is not None:
            mask_history = torch.zeros(patch_num, device=coord.device)
            mask_history[mask_patch_index] = 1.0
            self.vgc_diversity.update(mask_history)

        return point_mask, point_cluster

    @torch.no_grad()
    def _compute_cosine_difficulty(
        self, teacher_point, student_pred_sim, point_cluster, result_dict,
    ):
        """
        Compute per-token difficulty from teacher-student cosine similarity.

        t_i = teacher prototype logits (before SK sharpening)
        s_i = student prediction logits
        difficulty_i = 1 - cos(t_i, s_i)  ∈ [0, 2]

        High = student disagrees with teacher → hard region.
        Aggregated to grid patches for next forward's mask generation.
        """
        t_feat = teacher_point.feat.float()     # [N, 4096]
        s_feat = student_pred_sim.float()        # [N, 4096]

        cos_sim = F.cosine_similarity(t_feat, s_feat, dim=-1)  # [N]
        difficulty = 1.0 - cos_sim  # [N]

        # Aggregate to grid patches
        patch_num = point_cluster.max().item() + 1
        patch_diff = torch.zeros(patch_num, device=t_feat.device)
        patch_diff.scatter_add_(0, point_cluster, difficulty)
        patch_count = torch.zeros(patch_num, device=t_feat.device)
        patch_count.scatter_add_(0, point_cluster, torch.ones_like(difficulty))
        patch_diff = patch_diff / patch_count.clamp(min=1)

        self._cosine_patch_difficulty = patch_diff

        # Log metrics with more diagnostic detail
        if hasattr(self, "trainer") and self.trainer.writer is not None:
            step = self.trainer.writer.iter if hasattr(self.trainer.writer, 'iter') else 0
            w = self.trainer.writer
            w.add_scalar("cosine/difficulty_mean", difficulty.mean().item(), step)
            w.add_scalar("cosine/difficulty_std", difficulty.std().item(), step)
            w.add_scalar("cosine/cos_sim_mean", cos_sim.mean().item(), step)
            # Percentiles — if p10 stays low while p90 drops over time,
            # the metric is differentiating easy from hard regions
            d_sorted, _ = difficulty.sort()
            N = len(d_sorted)
            w.add_scalar("cosine/difficulty_p10", d_sorted[int(N * 0.1)].item(), step)
            w.add_scalar("cosine/difficulty_p50", d_sorted[int(N * 0.5)].item(), step)
            w.add_scalar("cosine/difficulty_p90", d_sorted[int(N * 0.9)].item(), step)
            w.add_scalar("cosine/p10_p90_gap",
                         d_sorted[int(N * 0.9)].item() - d_sorted[int(N * 0.1)].item(),
                         step)
            # Patch-level stats
            w.add_scalar("cosine/patch_difficulty_mean", patch_diff.mean().item(), step)
            w.add_scalar("cosine/patch_difficulty_std", patch_diff.std().item(), step)

            # Store difficulty for mask quality check (compare masked vs unmasked later)   
            if not hasattr(self, "_cosine_step"):
                self._cosine_step = 0
            self._cosine_step = step

    @torch.no_grad()
    def match_neighbour(
        self,
        view1_coord,
        view1_offset,
        view2_coord,
        view2_offset,
    ):
        index2, distance = pointops.knn_query(
            1,
            view2_coord.float(),
            view2_offset.int(),
            view1_coord.float(),
            view1_offset.int(),
        )
        index1 = torch.arange(
            index2.shape[0], device=index2.device, dtype=torch.long
        ).unsqueeze(-1)
        index = torch.cat([index1, index2], dim=-1)[
            distance.squeeze(-1) < self.match_max_r
        ]
        return index

    @torch.no_grad()
    def roll_point(self, point):
        n = self.num_global_view
        # [pc1, pc1', pc2, pc2'] -> [pc1', pc1, pc2', pc2], only support num_global_view == 2
        bs = len(point.offset) // self.num_global_view
        data_dict = {}
        for key in point.keys():
            if key in ["feat", "coord", "origin_coord", "batch"]:
                value = point[key].split(offset2bincount(point.offset).tolist())
                value = chain(*[value[n * b : n * (b + 1)][::-1] for b in range(bs)])
                if key == "batch":
                    value = [torch.ones_like(v) * i for i, v in enumerate(value)]
                data_dict[key] = torch.cat(list(value), dim=0)
        return Point(data_dict)

    def up_cast(self, point):
        for _ in range(self.up_cast_level):
            assert "pooling_parent" in point.keys()
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        return point

    def forward(self, data_dict, return_point=False):
        if return_point:
            point = self.teacher.backbone(data_dict)
            for _ in range(self.up_cast_level):
                assert "pooling_parent" in point.keys()
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            return dict(point=point)

        # prepare global_point, mask_global_point, local_point
        with torch.no_grad():
            # global_point & masking
            global_point = Point(
                feat=data_dict["global_feat"],
                coord=data_dict["global_coord"],
                origin_coord=data_dict["global_origin_coord"],
                offset=data_dict["global_offset"],
                grid_size=data_dict["grid_size"][0],
            )
            global_mask, global_cluster = self.generate_mask(
                global_point.coord, global_point.offset
            )
            # --- VGC mask overlap logging ---
            if self.masking_mode == "vgc" and self.vgc is not None:
                patch_num = global_cluster.max().item() + 1
                masked_patches = set(global_cluster[global_mask].unique().tolist())
                if self._prev_mask_patches is not None:
                    prev_set = self._prev_mask_patches
                    union = len(masked_patches | prev_set)
                    overlap = len(masked_patches & prev_set) / max(union, 1)
                    if (hasattr(self, "trainer") and self.trainer.writer is not None
                            and hasattr(self.trainer.writer, 'iter')):
                        s = self.trainer.writer.iter
                        self.trainer.writer.add_scalar("vgc/mask_overlap", overlap, s)
                        self.trainer.writer.add_scalar(
                            "vgc/mask_epoch_pct", len(masked_patches) / max(patch_num, 1), s)
                self._prev_mask_patches = masked_patches
            mask_global_coord = global_point.coord.clone().detach()
            if self.mask_jitter is not None:
                mask_global_coord[global_mask] += torch.clip(
                    torch.randn_like(mask_global_coord[global_mask]).mul(
                        self.mask_jitter
                    ),
                    max=self.mask_jitter * 2,
                )

            mask_global_point = Point(
                feat=data_dict["global_feat"],
                coord=mask_global_coord,
                origin_coord=data_dict["global_origin_coord"],
                mask=global_mask,
                offset=data_dict["global_offset"],
                grid_size=data_dict["grid_size"][0],
            )

            # local point & matching
            local_point = Point(
                feat=data_dict["local_feat"],
                coord=data_dict["local_coord"],
                origin_coord=data_dict["local_origin_coord"],
                offset=data_dict["local_offset"],
                grid_size=data_dict["grid_size"][0],
            )

            # create result dictionary for return
            result_dict = dict(loss=[])
            # teacher backbone forward (shared with mask and unmask)
            global_point_ = self.teacher.backbone(global_point)
            global_point_ = self.up_cast(global_point_)
            # teacher head forward
            # only use one shared head for both mask and unmask
            # priority: mask (global) > unmask (local)
            if self.mask_loss_weight > 0 or self.roll_mask_loss_weight > 0:
                global_point_.feat = self.teacher.mask_head(global_point_.feat)
            else:
                global_point_.feat = self.teacher.unmask_head(global_point_.feat)

        if self.mask_loss_weight > 0 or self.roll_mask_loss_weight > 0:
            # student forward
            mask_global_point_ = self.student.backbone(mask_global_point)
            mask_global_point_ = self.up_cast(mask_global_point_)
            mask_pred_sim = self.student.mask_head(mask_global_point_.feat)

            # --- Cosine difficulty: teacher-student agreement per token ---
            self._compute_cosine_difficulty(
                global_point_, mask_pred_sim, global_cluster, result_dict,
            )

            if self.mask_loss_weight > 0:
                with torch.no_grad():
                    match_index = self.match_neighbour(
                        mask_global_point_.origin_coord,
                        mask_global_point_.offset,
                        global_point_.origin_coord,
                        global_point_.offset,
                    )
                    # teacher forward
                    mask_target_sim = self.sinkhorn_knopp(
                        global_point_.feat[match_index[:, 1]],
                        self.teacher_temp,
                    )

                # loss
                mask_loss = -torch.sum(
                    mask_target_sim
                    * F.log_softmax(
                        mask_pred_sim[match_index[:, 0]] / self.student_temp, dim=-1
                    ),
                    dim=-1,
                )
                mask_loss = torch_scatter.segment_coo(
                    mask_loss,
                    index=mask_global_point_.batch[match_index[:, 0]],
                    reduce="mean",
                ).mean()
                result_dict["mask_loss"] = mask_loss
                result_dict["loss"].append(mask_loss * self.mask_loss_weight)

            if self.roll_mask_loss_weight > 0:
                roll_global_point_ = self.roll_point(global_point_)
                with torch.no_grad():
                    # match index for pred and roll target
                    match_index = self.match_neighbour(
                        mask_global_point_.origin_coord,
                        mask_global_point_.offset,
                        roll_global_point_.origin_coord,
                        roll_global_point_.offset,
                    )
                    # teacher forward
                    roll_mask_target_sim = self.sinkhorn_knopp(
                        roll_global_point_.feat[match_index[:, 1]],
                        self.teacher_temp,
                    )

                roll_mask_loss = -torch.sum(
                    roll_mask_target_sim
                    * F.log_softmax(
                        mask_pred_sim[match_index[:, 0]] / self.student_temp, dim=-1
                    ),
                    dim=-1,
                )
                roll_mask_loss = torch_scatter.segment_coo(
                    roll_mask_loss,
                    index=mask_global_point_.batch[match_index[:, 0]],
                    reduce="mean",
                ).mean()
                result_dict["roll_mask_loss"] = roll_mask_loss
                result_dict["loss"].append(roll_mask_loss * self.roll_mask_loss_weight)
        if self.unmask_loss_weight > 0:
            # student forward
            local_point_ = self.student.backbone(local_point)
            local_point_ = self.up_cast(local_point_)
            unmask_pred_sim = self.student.unmask_head(local_point_.feat)
            with torch.no_grad():
                principal_view_mask = global_point_.batch % self.num_global_view == 0
                principal_view_batch = (
                    global_point_.batch[principal_view_mask] // self.num_global_view
                )
                match_index = self.match_neighbour(
                    local_point_.origin_coord,
                    local_point_.offset[self.num_local_view - 1 :: self.num_local_view],
                    global_point_.origin_coord[principal_view_mask],
                    batch2offset(principal_view_batch),
                )
                # teacher forward
                unmask_target_sim = self.sinkhorn_knopp(
                    global_point_.feat[principal_view_mask][match_index[:, 1]],
                    self.teacher_temp,
                )
            # loss
            unmask_loss = -torch.sum(
                unmask_target_sim
                * F.log_softmax(
                    unmask_pred_sim[match_index[:, 0]] / self.student_temp, dim=-1
                ),
                dim=-1,
            )
            unmask_loss = torch_scatter.segment_coo(
                unmask_loss,
                index=local_point_.batch[match_index[:, 0]],
                reduce="mean",
            ).mean()
            result_dict["unmask_loss"] = unmask_loss
            result_dict["loss"].append(unmask_loss * self.unmask_loss_weight)

        # --- VGC classifier training (only when masking_mode == "vgc") ---
        # Train the classifier to predict teacher raw_cos_sim from backbone
        # features. Use features BEFORE the projection head (raw backbone output),
        # and target is the raw_cos_sim (softmax with high temperature).
        if self.vgc is not None:
            # Use teacher backbone features (before mask_head projection)
            # for classifier input. This ensures the classifier learns
            # from general-purpose features, not head-specific ones.
            with torch.no_grad():
                teacher_features = self.teacher.backbone(global_point)
                teacher_features = self.up_cast(teacher_features)

            # Teacher target: raw_cos_sim with soft target temperature
            raw_cos_sim = global_point_.feat  # after mask_head, before SK

            vgc_out = self.vgc(
                teacher_features.feat.unsqueeze(1),  # [N, 1, D]
                raw_cos_sim.unsqueeze(1),             # [N, 1, K]
            )
            vgc_loss = vgc_out["cls_loss"] * self.vgc_loss_weight
            result_dict["vgc_loss"] = vgc_loss
            result_dict["loss"].append(vgc_loss)
            result_dict["vgc_entropy"] = vgc_out["entropy"].mean()

        if self.vgc is not None:
            # --- VGC monitoring (only when writer is available) ---
            if hasattr(self, "trainer") and self.trainer.writer is not None:
                step = self.trainer.writer.iter if hasattr(self.trainer.writer, 'iter') else 0
                # Entropy statistics
                ent = vgc_out["entropy"]
                self.trainer.writer.add_scalar(
                    "vgc/entropy_mean", ent.mean().item(), step)
                self.trainer.writer.add_scalar(
                    "vgc/entropy_std", ent.std().item(), step)
                self.trainer.writer.add_scalar(
                    "vgc/entropy_max", ent.max().item(), step)
                self.trainer.writer.add_scalar(
                    "vgc/cls_loss", vgc_loss.item(), step)
                self.trainer.writer.add_scalar(
                    "vgc/alpha", self._vgc_alpha, step)
                # Classifier confidence histogram (bin counts)
                max_entropy = math.log(self.vgc.num_groups)
                if max_entropy > 0:
                    norm_ent = ent / max_entropy
                    for th in [0.1, 0.3, 0.5, 0.7, 0.9]:
                        ratio = (norm_ent < th).float().mean().item()
                        self.trainer.writer.add_scalar(
                            f"vgc/entropy_below_{int(th*100)}pct", ratio, step)

        result_dict["loss"] = sum(result_dict["loss"])

        if get_world_size() > 1:
            for loss in result_dict.values():
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        return result_dict
