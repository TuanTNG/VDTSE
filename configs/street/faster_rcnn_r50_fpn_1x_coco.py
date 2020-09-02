_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

load_from = '../checkpoints/pretrained/faster_rcnn_r50_fpn_2x_coco.pth'

model = dict(
    backbone=dict(
        frozen_stages=4,
        norm_cfg=dict(type='BN', requires_grad=False),
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=5,
        )
    )
)
# model training and testing settings


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
    samples_per_gpu=2,
    workers_per_gpu=4,
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

optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.0001)
