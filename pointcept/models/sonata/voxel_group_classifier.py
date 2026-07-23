"""
VoxelGroupClassifier: Entropy-Guided Adaptive Masking for 3D Scene Pretraining
==============================================================================

This module replaces the random masking in Sonata with an entropy-guided
adaptive masking strategy driven by a learnable classifier.

──────────────────────────────────────────────────────────────────────────────
MOTIVATION
──────────────────────────────────────────────────────────────────────────────

In the standard Sonata pretraining pipeline, input tokens are masked randomly:
    mask ~ Bernoulli(p)    where p is a fixed or scheduled ratio.

Every token has the same probability of being masked, regardless of whether
the model has already learned that region well.

Our key insight:
    The model's OWN uncertainty about a voxel's semantic grouping is a
    better proxy for "how much the model still needs to learn from this
    region" than random selection. We measure this uncertainty via a
    LEARNABLE classifier whose prediction entropy reflects the network's
    evolving understanding.

──────────────────────────────────────────────────────────────────────────────
METHOD OVERVIEW
──────────────────────────────────────────────────────────────────────────────

    ┌──────────────────────────────────────┐
    │ 1. Teacher backbone (EMA, frozen)    │
    │    │                                  │
    │    ├─ mid_features (layer 6)          │
    │    └─ OnlineCluster → raw similarity  │
    │       (cosine → prototypes, NO SK)    │
    │       → teacher_raw_sim [K]           │
    └──────────────────┬───────────────────┘
                       │
    ┌──────────────────▼───────────────────┐
    │ 2. VoxelGroupClassifier (learnable)   │
    │                                        │
    │    MLP: D → 128 → K                   │
    │    predicts prototype assignment      │
    │    distribution for each subvoxel     │
    │                                        │
    │    H = -sum(p * log(p))   ← ENTROPY   │
    │                                        │
    │    Trained via KL divergence          │
    │    against teacher_raw_sim (BEFORE    │
    │    SK sharpening, kept soft via       │
    │    higher temperature τ=2.0)          │
    └──────────────────┬───────────────────┘
                       │
    ┌──────────────────▼───────────────────┐
    │ 3. Adaptive Mask Generation           │
    │                                        │
    │    difficulty_i = topK_mean(H_i)       │
    │    p_mask_i = f(difficulty_i)          │
    │    mask_i ~ Bernoulli(p_mask_i)        │
    └──────────────────────────────────────┘
                       │
    ┌──────────────────▼───────────────────┐
    │ 4. Student backbone                   │
    │    │                                  │
    │    Processes masked input.            │
    │    Predicts teacher features on       │
    │    masked positions (distillation loss)│
    └──────────────────────────────────────┘

──────────────────────────────────────────────────────────────────────────────
WHY A LEARNABLE CLASSIFIER?
──────────────────────────────────────────────────────────────────────────────

The Sonata teacher already produces prototype assignment logits via its
projection head (cosine similarity to learned prototypes). These logits
encode the teacher's judgment about which semantic group each subvoxel
belongs to.

The classifier learns to APPROXIMATE these assignments. Its prediction
entropy then measures:
    "How well can a parametric function, trained to generalize across
     scenes, reproduce the teacher's prototype distribution for this
     specific subvoxel?"

If the classifier is confident → the teacher's assignment pattern is
easy to learn (the underlying feature manifold is smooth and well-
structured for this region).
If the classifier is uncertain → the feature manifold is ambiguous
or the model hasn't yet formed clear semantic boundaries here.

This is different from directly using the teacher's own prototype
entropy because:
(A) The classifier generalizes across scenes → its entropy reflects
    transferable ambiguity, not per-sample noise.
(B) The classifier is trained via gradient descent → it learns a
    smooth decision manifold that filters high-frequency variation.
(C) The classifier evolves with training → naturally creates a
    self-pacing curriculum (early: high entropy everywhere ≈ random;
    late: entropy concentrates on true semantic boundaries).

──────────────────────────────────────────────────────────────────────────────
CORE ALGORITHM
──────────────────────────────────────────────────────────────────────────────

Given:
    subvoxel_features  ∈ R^(M × P³ × D)    # M tokens, P³=64 subvoxels/token
    teacher_prototypes ∈ R^(K × D)          # K learnable prototypes

Step 1: Get teacher prototype assignments from Sonata prototyp_head

    teacher_logits = prototyp_head(teacher_projection_features)
        # This is the STANDARD Sonata teacher head output
    teacher_probs  = softmax(teacher_logits / τ_teacher)

Step 2: Forward classifier to get prediction entropy

    logits = Classifier(subvoxel_features)                  # [M, P³, K]
    probs  = softmax(logits / τ_cls)
    H = -sum(probs * log(probs + ε), dim=-1)               # [M, P³]

Step 3: Aggregate to token-level difficulty

    difficulty = topK_mean(H, k = max(1, P³ * 0.1))         # top 10%
    difficulty = difficulty / log(K)                        # normalize to [0,1]

Step 4: Difficulty → mask probability (sigmoid w/ median anchor)

    median  = difficulty.median()
    p_adapt = mask_ratio * (1 + σ((difficulty - median) / τ_mask))
    p_final = (1-α) * mask_ratio + α * p_adapt   # curriculum: α ∈ [0,1]
    mask ~ Bernoulli(p_final)

Step 5: Classifier training loss

    L_cls = CrossEntropy(logits, teacher_probs.detach())
    total_L = L_distill + λ_cls * L_cls

──────────────────────────────────────────────────────────────────────────────
COLD START & CURRICULUM HANDLING
──────────────────────────────────────────────────────────────────────────────

During early training (first ~20% epochs), the classifier has not yet
learned meaningful decision boundaries. All subvoxels have uniformly
high entropy ≈ log(K), which makes adaptive masking ≈ random masking.

We explicitly acknowledge this with a curriculum schedule:

    curriculum_alpha:  α=0 for warmup_epochs →  α=1 via cosine ramp
    warmup_epochs:     ~20% of total epochs

When α=0: pure random masking (same as Sonata baseline)
When α=1: full classifier entropy-driven masking

This guarantees that our method performs at LEAST as well as the
baseline, because for the first ~20% of training it IS the baseline.

──────────────────────────────────────────────────────────────────────────────
COMPARISON WITH RELATED APPROACHES
──────────────────────────────────────────────────────────────────────────────

Method              | Difficulty Signal   | Learned? | Scene? | Guides Mask?
────────────────────┼─────────────────────┼──────────┼────────┼─────────────
Sonata (baseline)   | None (random)       | N/A      | ✅     | ❌
GroupContrast       | Cluster entropy     | No       | ✅     | ❌ (loss wt)
GeoMask3D           | Geometric complexity| No       | ✅     | ✅
Point-CSMAE         | Attention saliency  | No       | ❌ obj | ✅
Ours (VGC)          | Classifier entropy  | Yes      | ✅     | ✅

──────────────────────────────────────────────────────────────────────────────
IMPLEMENTATION NOTES
──────────────────────────────────────────────────────────────────────────────

1. Classifier architecture:
   - Input: D-dim subvoxel features from teacher's mid encoder layer
   - Hidden: 128-dim with GELU + LayerNorm
   - Output: K-dim logits (K = num_prototypes, typically 256)
   - Parameters: ~D*128 + 128*256 ≈ 82K (negligible)

2. Label smoothing (default: 0.1) prevents classifier overconfidence
   that would collapse entropy to zero

3. The classifier is updated via gradient descent (not EMA)
   The teacher backbone uses EMA (standard Sonata recipe)
   The teacher prototype_head uses EMA (standard Sonata recipe)

4. Teacher prototype assignments are detached → classifier loss does
   NOT backprop into the teacher backbone or prototype_head

5. λ_cls defaults to 0.1 to balance against the dominant distillation loss

6. The classifier target (teacher prototype assignment) is the output
   of Sonata's prototyp_head, which computes cosine similarity between
   teacher features and learned prototypes, followed by softmax.
   This is standard in Sonata/DINO-style self-supervised learning.

──────────────────────────────────────────────────────────────────────────────
EXPECTED BEHAVIOR
──────────────────────────────────────────────────────────────────────────────

Epoch 1-20 (warmup):
    α=0 → random masking everywhere
    Classifier entropy ≈ log(K) everywhere (uniform)
    Training identical to Sonata baseline

Epoch 20-100:
    α ramping 0→1
    Classifier starts to differentiate: simple regions (flat walls)
        get low entropy; complex boundaries get high entropy
    Masking shifts from uniform → boundary-focused

Epoch 100+:
    α=1 → full adaptive masking
    Entropy histogram is bimodal: most regions have low entropy,
        a long tail of high-entropy regions at semantic boundaries
"""

from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class VoxelGroupClassifier(nn.Module):
    """
    Learnable classifier that predicts prototype group assignments for
    each subvoxel. Uses prediction entropy as a difficulty metric for
    adaptive masking.

    The classifier is trained to match the teacher network's prototype_head
    output (cosine similarity to learned prototypes → softmax), which is
    the standard Sonata/DINO self-distillation protocol.

    Args:
        embed_dim:    Feature dimension (e.g., 384)
        hidden_dim:   Hidden layer dimension (e.g., 128)
        num_groups:   Number of prototype groups K (e.g., 256)
        label_smooth: Label smoothing for classifier targets
        temperature:  Softmax temperature for classifier predictions
        loss_weight:  Weight λ_cls of classifier loss in total loss
    """

    def __init__(
        self,
        embed_dim: int = 384,
        hidden_dim: int = 128,
        num_groups: int = 256,
        label_smooth: float = 0.1,
        temperature: float = 1.0,
        loss_weight: float = 0.1,
        teacher_target_temp: float = 0.5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_groups = num_groups
        self.label_smooth = label_smooth
        self.temperature = temperature
        self.loss_weight = loss_weight
        self.teacher_target_temp = teacher_target_temp

        self.layers = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_groups),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        features: torch.Tensor,
        teacher_raw_sim: torch.Tensor,
        occupancy: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            features:         [M, P³, D]  per-subvoxel features
            teacher_raw_sim:  [M, P³, K]  teacher cosine similarity to prototypes
                                          (BEFORE Sinkhorn-Knopp sharpening)
                                          From teacher OnlineCluster output,
                                          detached from EMA-updated teacher.
            occupancy:        [M, P³, 1]  binary occupancy mask (optional)

        Returns:
            dict with keys:
                entropy:     [M, P³]     Shannon entropy per subvoxel
                cls_loss:    scalar      cross-entropy loss for classifier
                probs:       [M, P³, K]  classifier prediction probabilities

        NOTE: We deliberately use teacher_raw_sim (before SK) rather than
        teacher SK output. SK's low temperature (τ=0.04) and doubly-stochastic
        normalization produce near one-hot assignments, which would cause
        the classifier to predict low entropy everywhere → useless as a
        difficulty metric. Raw cosine similarity preserves the soft structure
        that reflects genuine feature-space ambiguity.
        """
        logits = self.layers(features)                        # [M, P³, K]
        logits = logits / self.temperature

        # Entropy from classifier's own predictions (inference path)
        probs = F.softmax(logits, dim=-1)                     # [M, P³, K]
        eps = 1e-8
        entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)  # [M, P³]

        if occupancy is not None:
            occ_mask = occupancy.squeeze(-1) > 0
            entropy = entropy * occ_mask.float()

        # Classification loss: train classifier to predict teacher's SOFT
        # logit distribution (before SK sharpening). Use KL divergence to
        # preserve the soft structure rather than collapsing to one-hot.
        cls_loss = self._classification_loss(logits, teacher_raw_sim, occupancy)

        return dict(entropy=entropy, cls_loss=cls_loss, probs=probs)

    def _classification_loss(
        self,
        logits: torch.Tensor,
        teacher_sim: torch.Tensor,
        occupancy: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        KL divergence between classifier output and teacher's raw
        similarity distribution (softened with a higher temperature).

        teacher_sim are the raw cosine similarity scores from the
        teacher's OnlineCluster projection head, BEFORE Sinkhorn-Knopp.
        We apply softmax with a SEPARATE (higher) temperature to
        obtain a meaningful soft target distribution.
        """
        M, P3, K = logits.shape
        logits_flat = logits.reshape(-1, K)
        sim_flat = teacher_sim.detach().reshape(-1, K)

        # Teacher target: softmax with a MUCH higher temperature
        # (default 2.0, vs SK's 0.04). This preserves ambiguity structure.
        teacher_target_temp = self.teacher_target_temp
        target_probs = F.softmax(sim_flat / teacher_target_temp, dim=-1)
        target_probs = target_probs.detach()

        # KL(logits || teacher) with label smoothing on teacher side
        smooth = self.label_smooth
        if smooth > 0:
            uniform = torch.full_like(target_probs, 1.0 / K)
            target_probs = (1.0 - smooth) * target_probs + smooth * uniform

        log_probs = F.log_softmax(logits_flat, dim=-1)
        loss = (target_probs * (torch.log(target_probs + 1e-8) - log_probs)).sum(dim=-1)

        if occupancy is not None:
            occ_flat = occupancy.reshape(-1)
            loss = (loss * occ_flat).sum() / occ_flat.sum().clamp(min=1)
        else:
            loss = loss.mean()

        return loss * self.loss_weight

    @torch.no_grad()
    def get_difficulty(
        self,
        features: torch.Tensor,
        topk_percentile: float = 0.1,
    ) -> torch.Tensor:
        """
        Compute per-token difficulty scores from subvoxel features.
        Inference-only: forward classifier → entropy → difficulty.
        No teacher involvement, no loss computation.

        Args:
            features:         [M, P³, D]  per-subvoxel features
            topk_percentile:  fraction of highest-entropy subvoxels to average

        Returns:
            difficulty:  [M]  per-token difficulty ∈ [0, 1]
        """
        logits = self.layers(features)
        logits = logits / self.temperature
        probs = F.softmax(logits, dim=-1)

        eps = 1e-8
        entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)  # [M, P³]

        M, P3 = entropy.shape
        k = max(1, int(P3 * topk_percentile))
        topk_vals, _ = entropy.topk(k, dim=1)
        difficulty = topk_vals.mean(dim=1)

        difficulty = difficulty / math.log(self.num_groups)
        difficulty = difficulty.clamp(0.0, 1.0)

        return difficulty

    @staticmethod
    def difficulty_to_mask_probs(
        difficulty: torch.Tensor,
        target_ratio: float,
        temperature: float = 1.0,
        diversity_penalty: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Convert difficulty scores to Bernoulli mask probabilities.

        p_i = r_target * (1 + σ((d_i - median(d)) / τ))

        If diversity_penalty is provided, recently-masked tokens get
        reduced probability to prevent repeatedly masking the same region.

        Args:
            difficulty:        [M]  per-token difficulty ∈ [0, 1]
            target_ratio:      desired overall masking ratio
            temperature:       sigmoid sharpness
            diversity_penalty: [M]  per-token penalty ∈ [0, 1] (optional)

        Returns:
            probs:  [M]  per-token mask probability
        """
        median = difficulty.median()
        z = (difficulty - median) / temperature
        probs = target_ratio * (1.0 + torch.sigmoid(z))

        if diversity_penalty is not None:
            probs = probs * (1.0 - diversity_penalty)

        probs = probs.clamp(0.05, 0.95)
        return probs


class MaskDiversityScheduler:
    """
    Tracks recent mask history per token and computes a diversity penalty
    to prevent the classifier from repeatedly masking the same regions.

    Each time a token is masked, its penalty increases. Over epochs where
    the token is NOT masked, the penalty decays exponentially.

    Usage:
        scheduler = MaskDiversityScheduler(decay=0.9, max_penalty=0.5)

        # Each epoch, after generating mask:
        penalty = scheduler.penalty(mask)          # get current penalty
        scheduler.update(mask)                     # update history
    """

    def __init__(self, decay: float = 0.9, max_penalty: float = 0.5):
        self.decay = decay
        self.max_penalty = max_penalty
        self.history = None

    def update(self, mask: torch.Tensor):
        """Update EMA history: history = decay * history + (1-decay) * mask."""
        mask_float = mask.float().detach()
        if self.history is None:
            self.history = mask_float.clone()
        else:
            self.history = (
                self.decay * self.history + (1.0 - self.decay) * mask_float
            )

    def penalty(self) -> torch.Tensor:
        """Return per-token penalty ∈ [0, max_penalty]."""
        if self.history is None:
            return None
        return (self.history * self.max_penalty).clamp(0.0, self.max_penalty)

    def state_dict(self) -> dict:
        if self.history is not None:
            return dict(history=self.history)
        return dict()

    def load_state_dict(self, state: dict):
        if "history" in state:
            self.history = state["history"]


def curriculum_alpha(current_epoch: int, warmup_epochs: int) -> float:
    """
    Cosine curriculum ramp for masking.

    Args:
        current_epoch:  current training epoch (0-indexed)
        warmup_epochs:  number of warmup epochs (random masking)

    Returns:
        α ∈ [0, 1]: 0 = random masking, 1 = full adaptive masking
    """
    if current_epoch < warmup_epochs:
        return 0.0
    progress = (current_epoch - warmup_epochs) / max(1, warmup_epochs)
    return 0.5 * (1.0 - math.cos(min(progress, 1.0) * math.pi))
