_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py', 
    '../_base_/default_runtime.py',
    '../_base_/models/retinanet_r50_fpn.py'
]
load_from = '../checkpoints/pretrained/retinanet_x101_32x4d_fpn_1x_coco.pth'

model = dict(
    pretrained='open-mmlab://resnext101_32x4d',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'),
)

classes = ('motorbike','car','bus','truck','person',)
DATA_ROOT = 'data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

data = dict(
    # samples_per_gpu=8,
    # workers_per_gpu=4,
    train=dict(
        ann_file=DATA_ROOT + 'annotations/train.json',
        img_prefix=DATA_ROOT + 'images/full/full/',
        classes=classes,
        pipeline=train_pipeline
    ),
    val=dict(
        ann_file=DATA_ROOT + 'annotations/test.json',
        img_prefix=DATA_ROOT + 'images/full/full/',
        classes=classes,
    ),
    test=dict(
        ann_file=DATA_ROOT + 'annotations/test.json',
        img_prefix=DATA_ROOT + 'images/full/full/',
        classes=classes,
    ))

# optimizer
optimizer = dict(type='SGD', lr=1e-2, momentum=0.9, weight_decay=0.0001)