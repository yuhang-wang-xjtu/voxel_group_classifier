"""
Minimal smoke test config for VoxelGroupClassifier.
Purpose: verify code runs, mask signal works, gradients flow.
Dataset: S3DIS Area_1 only (fastest loading)
Epochs: 10
"""
_base_ = ["../sonata/pretrain-sonata-v1m2-0-uni-teacher-head.py"]

# Cut training to absolute minimum
epoch = 10
batch_size = 8      # single GPU, small batch
num_worker = 4
evaluate = False
empty_cache = True
enable_amp = True

# Override model — same as full config but with VGC + debug mode
model = dict(
    _delete_=True,
    type="Sonata-v1m2",
    backbone=dict(
        type="PT-v3m2",
        in_channels=9,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=True,
        enc_mode=True,
        mask_token=True,
    ),
    teacher_custom=dict(attn_drop=0.0, proj_drop=0.0, drop_path=0.0),

    head_in_channels=1088,
    head_hidden_channels=4096,
    head_embed_channels=256,
    head_num_prototypes=4096,
    num_global_view=2,
    num_local_view=4,
    mask_size_start=0.1,
    mask_size_base=0.2,
    mask_ratio_start=0.3,
    mask_ratio_base=0.5,
    mask_jitter=0.01,
    teacher_temp_start=0.04,
    teacher_temp_base=0.07,
    student_temp=0.1,
    mask_loss_weight=2 / 8,
    roll_mask_loss_weight=2 / 8,
    unmask_loss_weight=4 / 8,
    momentum_base=0.994,
    momentum_final=1,
    match_max_k=8,
    match_max_r=0.32,
    up_cast_level=2,

    # === VGC (entropy-guided) ===
    masking_mode="vgc",
    vgc_embed_dim=512,
    vgc_hidden_dim=128,
    vgc_num_groups=4096,
    vgc_label_smooth=0.1,
    vgc_temperature=1.0,
    vgc_target_temp=2.0,
    vgc_loss_weight=0.1,
    vgc_warmup_ratio=0.0,     # warmup=0 → adaptive from step 1
    vgc_diversity_decay=0.9,
    vgc_diversity_max_penalty=0.5,
)

# S3DIS Area_1 only — ~40 rooms, <1 GB
data = dict(
    train=dict(
        type="ConcatDataset",
        datasets=[
            dict(
                type="S3DISDataset",
                split=["Area_1"],
                data_root="data/s3dis",
                transform="${transform}",
                test_mode=False,
                loop=1,
            ),
        ],
    )
)

# Disable wandb
enable_wandb = False

# Scheduler — shorter warmup
optimizer = dict(type="AdamW", lr=2e-4, weight_decay=0.04)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[2e-4],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# Save checkpoint at epoch 5 and 10
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="WeightDecaySchedular", base_value=0.04, final_value=0.2),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=5),
]
