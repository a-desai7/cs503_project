norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='DTP',
    use_depth=True,
    pretrained=None,
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=192,
        embed_dims=128,
        patch_size=4,
        window_size=6,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'checkpoints/simmim_pretrain__swin_base__img192_window6__800ep.pth'
        )),
    decode_head=dict(
        type='IAParser',
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            avg_non_ignore=True),
        illumination_channels=128,
        illumination_features_channels=64),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 1024), stride=(341, 683)),
    disturb_beta='uniform',
    disentangle_loss=dict(type='PixelLoss', loss_weight=0.1, loss_type='L1'),
    disentangle_head=dict(
        type='SODHead',
        channels=32,
        in_channels=5,
        ill_embeds_op='-',
        clip=False,
        norm_cfg=dict(type='IN2d', requires_grad=True),
        init_cfg=dict(
            type='Kaiming',
            override=dict(
                type='Constant',
                layer='Conv2d',
                name='reflectance_output',
                val=0.0,
                bias=0.5)),
        loss_smooth=dict(type='SmoothLoss', loss_weight=0.01),
        loss_retinex=dict(type='PixelLoss', loss_weight=1.0, loss_type='L2')))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDepthFromFile', depth_folder='depth_worse'),
    dict(type='ConcatDepth'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(256, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(256, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDepthFromFile', depth_folder='depth_worse'),
    dict(type='ConcatDepth'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
nightlab_train = dict(
    type='NightcityDataset',
    data_root='data/nightcity-fine/train',
    img_dir='img',
    ann_dir='lbl',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadDepthFromFile', depth_folder='depth_worse'),
        dict(type='ConcatDepth'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(1024, 512), ratio_range=(0.5, 2.0)),
        dict(type='RandomCrop', crop_size=(256, 512), cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size=(256, 512), pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])
    ])
nightlab_test = dict(
    type='NightcityDataset',
    data_root='data/nightcity-fine/val',
    img_dir='img',
    ann_dir='lbl',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadDepthFromFile', depth_folder='depth_worse'),
        dict(type='ConcatDepth'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(1024, 512),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ])
    ])
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='DTPDataset',
        datasetA=dict(
            type='NightcityDataset',
            data_root='data/nightcity-fine/train',
            img_dir='img',
            ann_dir='lbl',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadDepthFromFile', depth_folder='depth_worse'),
                dict(type='ConcatDepth'),
                dict(type='LoadAnnotations'),
                dict(
                    type='Resize',
                    img_scale=(1024, 512),
                    ratio_range=(0.5, 2.0)),
                dict(
                    type='RandomCrop',
                    crop_size=(256, 512),
                    cat_max_ratio=0.75),
                dict(type='RandomFlip', prob=0.5),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size=(256, 512), pad_val=0, seg_pad_val=255),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'])
            ]),
        datasetB=dict(
            type='NightcityDataset',
            data_root='data/nightcity-fine/train',
            img_dir='img',
            ann_dir='lbl',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadDepthFromFile', depth_folder='depth_worse'),
                dict(type='ConcatDepth'),
                dict(type='LoadAnnotations'),
                dict(
                    type='Resize',
                    img_scale=(1024, 512),
                    ratio_range=(0.5, 2.0)),
                dict(
                    type='RandomCrop',
                    crop_size=(256, 512),
                    cat_max_ratio=0.75),
                dict(type='RandomFlip', prob=0.5),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size=(256, 512), pad_val=0, seg_pad_val=255),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'])
            ])),
    val=dict(
        type='NightcityDataset',
        data_root='data/nightcity-fine/val',
        img_dir='img',
        ann_dir='lbl',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadDepthFromFile', depth_folder='depth_worse'),
            dict(type='ConcatDepth'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='NightcityDataset',
        data_root='data/nightcity-fine/val',
        img_dir='img',
        ann_dir='lbl',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadDepthFromFile', depth_folder='depth_worse'),
            dict(type='ConcatDepth'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=3)
evaluation = dict(
    interval=2000, metric='mIoU', pre_eval=True, save_best='mIoU')
checkpoint_file = 'checkpoints/simmim_pretrain__swin_base__img192_window6__800ep.pth'
fp16 = dict()
find_unused_parameters = True
gpu_ids = range(0, 2)
load_from = None
resume_from = 'work_dirs/night_depth_worse/latest.pth'
auto_resume = False
work_dir = './work_dirs/night_depth_worse'
