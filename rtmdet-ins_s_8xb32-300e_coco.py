default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=6),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = '/content/rtmdet-ins_s_8xb32-300e_coco_20221121_212604-fdc5d7ec.pth'
resume = False
work_dir = './experiments'
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=30,
    val_interval=2,
    dynamic_intervals=[(28, 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-05, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0002,
        begin=15,
        end=30,
        T_max=15,
        by_epoch=True,
        convert_to_iter_based=True)
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0005, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))
auto_scale_lr = dict(enable=False, base_batch_size=16)
dataset_type = 'CocoDataset'
data_root = './'
file_client_args = dict(backend='disk')
metainfo = dict(
    classes=('pants', 'shirt, blouse', 'top, t-shirt, sweatshirt', 'jacket',
             'dress', 'glasses', 'shoe', 'bag, wallet', 'hat', 'watch'),
    pallete=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
             (106, 0, 228), (0, 60, 100), (0, 80, 100),
             (0, 0, 70), (0, 0, 192), (250, 170, 30), (100, 170, 30),
             (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
             (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0),
             (120, 166, 157), (110, 76, 0), (174, 57, 255), (199, 100, 0),
             (72, 0, 118), (255, 179, 240), (0, 125, 92), (209, 0, 151),
             (188, 208, 182), (0, 220, 176), (255, 99, 164), (92, 0, 73),
             (133, 129, 255), (78, 180, 255), (0, 228, 0), (174, 255, 243),
             (45, 89, 255), (134, 134, 103), (145, 148, 174), (255, 208, 186),
             (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
             (151, 0, 95)])
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='CachedMosaic', img_scale=(640, 640), pad_val=114.0),
    dict(
        type='RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=(640, 640),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='CachedMixUp',
        img_scale=(640, 640),
        ratio_range=(1.0, 1.0),
        max_cached_images=20,
        pad_val=(114, 114, 114)),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=5,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type='CocoDataset',
        data_root='./',
        ann_file='instances_attributes_val2020.json',
        data_prefix=dict(img='dataset/test/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(
            classes=('pants', 'shirt, blouse', 'top, t-shirt, sweatshirt',
                     'jacket', 'dress', 'glasses', 'shoe', 'bag, wallet',
                     'hat', 'watch'),
            pallete=[
                (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
                (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
                (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
                (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
                (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
                (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
                (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
                (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
                (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
                (134, 134, 103), (145, 148, 174), (255, 208, 186),
                (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
                (151, 0, 95)
            ]),
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True,
                poly2mask=False),
            dict(type='CachedMosaic', img_scale=(640, 640), pad_val=114.0),
            dict(
                type='RandomResize',
                scale=(1280, 1280),
                ratio_range=(0.5, 2.0),
                keep_ratio=True),
            dict(
                type='RandomCrop',
                crop_size=(640, 640),
                recompute_bbox=True,
                allow_negative_crop=True),
            dict(type='YOLOXHSVRandomAug'),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='Pad', size=(640, 640),
                pad_val=dict(img=(114, 114, 114))),
            dict(
                type='CachedMixUp',
                img_scale=(640, 640),
                ratio_range=(1.0, 1.0),
                max_cached_images=20,
                pad_val=(114, 114, 114)),
            dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
            dict(type='PackDetInputs')
        ]),
    pin_memory=True)
val_dataloader = dict(
    batch_size=5,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='./',
        ann_file='instances_attributes_val2020.json',
        data_prefix=dict(img='dataset/test/'),
        test_mode=True,
        metainfo=dict(
            classes=('pants', 'shirt, blouse', 'top, t-shirt, sweatshirt',
                     'jacket', 'dress', 'glasses', 'shoe', 'bag, wallet',
                     'hat', 'watch'),
            pallete=[
                (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
                (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
                (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
                (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
                (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
                (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
                (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
                (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
                (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
                (134, 134, 103), (145, 148, 174), (255, 208, 186),
                (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
                (151, 0, 95)
            ]),
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='Resize', scale=(640, 640), keep_ratio=True),
            dict(
                type='Pad', size=(640, 640),
                pad_val=dict(img=(114, 114, 114))),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ]))
test_dataloader = dict(
    batch_size=5,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='./',
        ann_file='instances_attributes_val2020.json',
        data_prefix=dict(img='dataset/test/'),
        test_mode=True,
        metainfo=dict(
            classes=('pants', 'shirt, blouse', 'top, t-shirt, sweatshirt',
                     'jacket', 'dress', 'glasses', 'shoe', 'bag, wallet',
                     'hat', 'watch'),
            pallete=[
                (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
                (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
                (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
                (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
                (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
                (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
                (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
                (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
                (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
                (134, 134, 103), (145, 148, 174), (255, 208, 186),
                (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
                (151, 0, 95)
            ]),
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='Resize', scale=(640, 640), keep_ratio=True),
            dict(
                type='Pad', size=(640, 640),
                pad_val=dict(img=(114, 114, 114))),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ]))
val_evaluator = dict(
    type='CocoMetric',
    ann_file='/content/instances_attributes_val2020.json',
    metric=['bbox', 'segm'],
    format_only=False,
    proposal_nums=(100, 1, 10))
test_evaluator = dict(
    type='CocoMetric',
    ann_file='/content/instances_attributes_val2020.json',
    metric=['bbox', 'segm'],
    format_only=False,
    proposal_nums=(100, 1, 10))
tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.6), max_per_img=100))
img_scales = [(640, 640), (320, 320), (960, 960)]
tta_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(
        type='TestTimeAug',
        transforms=[[{
            'type': 'Resize',
            'scale': (640, 640),
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale': (320, 320),
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale': (960, 960),
            'keep_ratio': True
        }],
                    [{
                        'type': 'RandomFlip',
                        'prob': 1.0
                    }, {
                        'type': 'RandomFlip',
                        'prob': 0.0
                    }],
                    [{
                        'type': 'Pad',
                        'size': (960, 960),
                        'pad_val': {
                            'img': (114, 114, 114)
                        }
                    }],
                    [{
                        'type':
                        'PackDetInputs',
                        'meta_keys':
                        ('img_id', 'img_path', 'ori_shape', 'img_shape',
                         'scale_factor', 'flip', 'flip_direction')
                    }]])
]
model = dict(
    type='RTMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        batch_augments=None),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.33,
        widen_factor=0.5,
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint=
            'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth'
        )),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='RTMDetInsSepBNHead',
        num_classes=46,
        in_channels=128,
        stacked_convs=2,
        share_conv=True,
        pred_kernel_size=1,
        feat_channels=128,
        act_cfg=dict(type='SiLU', inplace=True),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        anchor_generator=dict(
            type='MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_mask=dict(
            type='DiceLoss', loss_weight=2.0, eps=5e-06, reduction='mean')),
    train_cfg=dict(
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100,
        mask_thr_binary=0.5))
train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=(640, 640),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]
max_epochs = 30
stage2_num_epochs = 5
base_lr = 0.0005
interval = 10
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=280,
        switch_pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True,
                poly2mask=False),
            dict(
                type='RandomResize',
                scale=(640, 640),
                ratio_range=(0.5, 2.0),
                keep_ratio=True),
            dict(
                type='RandomCrop',
                crop_size=(640, 640),
                recompute_bbox=True,
                allow_negative_crop=True),
            dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
            dict(type='YOLOXHSVRandomAug'),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='Pad', size=(640, 640),
                pad_val=dict(img=(114, 114, 114))),
            dict(type='PackDetInputs')
        ])
]
checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth'
launcher = 'none'
