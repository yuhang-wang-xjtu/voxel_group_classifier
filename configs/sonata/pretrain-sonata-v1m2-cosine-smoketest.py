"""Minimal cosine masking smoke test config (S3DIS, 10 epoch)."""
_base_ = ["../sonata/pretrain-sonata-v1m2-0-uni-teacher-head.py"]

epoch = 10
eval_epoch = 10
batch_size = 8
num_worker = 4
evaluate = False
enable_wandb = False

model = dict(
    _delete_=True,
    type="Sonata-v1m2-vgc",
    backbone=dict(
        type="PT-v3m2",
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        mlp_ratio=4, qkv_bias=True, qk_scale=None,
        attn_drop=0.0, proj_drop=0.0, drop_path=0.0,
        shuffle_orders=True, pre_norm=True,
        enable_rpe=False, enable_flash=True,
        upcast_attention=False, upcast_softmax=False,
        traceable=True, enc_mode=True, mask_token=True,
    ),
    teacher_custom=dict(attn_drop=0.0, proj_drop=0.0, drop_path=0.0),
    head_in_channels=1088, head_hidden_channels=4096,
    head_embed_channels=256, head_num_prototypes=4096,
    num_global_view=2, num_local_view=4,
    mask_size_start=0.1, mask_size_base=0.2,
    mask_ratio_start=0.3, mask_ratio_base=0.5,
    mask_jitter=0.01,
    teacher_temp_start=0.04, teacher_temp_base=0.07,
    student_temp=0.1,
    mask_loss_weight=2/8, roll_mask_loss_weight=2/8, unmask_loss_weight=4/8,
    momentum_base=0.994, momentum_final=1,
    match_max_k=8, match_max_r=0.32, up_cast_level=2,
    # cosine masking
    masking_mode="cosine",
    vgc_warmup_ratio=0.0,
)

data = dict(train=dict(type="ConcatDataset", datasets=[
    dict(type="S3DISDataset", split=["Area_1"], data_root="data/s3dis",
         test_mode=False, loop=1)
]))


# Override transform: remove normal dependency (HDF5 data may not have normals)
transform = [
    dict(type='GridSample', grid_size=0.02, hash_type='fnv', mode='train'),
    dict(type='Copy', keys_dict=dict(coord='origin_coord')),
    dict(
        type='MultiViewGenerator',
        view_keys=('coord', 'origin_coord', 'color'),
        global_view_num=2,
        global_view_scale=(0.4, 1.0),
        local_view_num=4,
        local_view_scale=(0.1, 0.4),
        global_shared_transform=[
            dict(type='RandomColorJitter', brightness=0.4, contrast=0.4, saturation=0.2, hue=0.02, p=0.8),
            dict(type='ChromaticTranslation', p=0.95, ratio=0.05),
            dict(type='NormalizeColor')
        ],
        global_transform=[
            dict(type='CenterShift', apply_z=True),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='RandomRotate', angle=[-1, 1], axis='z', center=[0, 0, 0], p=0.8),
            dict(type='RandomFlip', p=0.5),
            dict(type='RandomJitter', sigma=0.005, clip=0.02),
            dict(type='ElasticDistortion', distortion_params=[[0.2, 0.4], [0.8, 1.6]])
        ],
        local_transform=[
            dict(type='CenterShift', apply_z=True),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='RandomRotate', angle=[-1, 1], axis='z', center=[0, 0, 0], p=0.8),
            dict(type='RandomFlip', p=0.5),
            dict(type='RandomJitter', sigma=0.005, clip=0.02),
            dict(type='ElasticDistortion', distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type='RandomColorJitter', brightness=0.4, contrast=0.4, saturation=0.2, hue=0.02, p=0.8),
            dict(type='ChromaticTranslation', p=0.95, ratio=0.05),
            dict(type='NormalizeColor')
        ],
        max_size=65536),
    dict(type='ToTensor'),
    dict(type='Update', keys_dict=dict(grid_size=0.02)),
    dict(
        type='Collect',
        keys=('global_origin_coord', 'global_coord', 'global_color',
              'global_offset', 'local_origin_coord', 'local_coord',
              'local_color', 'local_offset', 'grid_size', 'name'),
        offset_keys_dict=dict(),
        global_feat_keys=('global_coord', 'global_color'),  # no normal
        local_feat_keys=('local_coord', 'local_color'))     # no normal
]

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="WeightDecaySchedular", base_value=0.04, final_value=0.2),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=5),
]

