model = dict(
    type='SkeletonGCN',
    backbone=dict(
        type='STGCN',
        in_channels=3,
        edge_importance_weighting=True,
        graph_cfg=dict(layout='ntu-rgb+d', strategy='spatial')),
        # graph_cfg=dict(layout='coco', strategy='spatial')),
    cls_head=dict(
        type='STGCNHead',
        num_classes=60,
        in_channels=256,
        loss_cls=dict(type='CrossEntropyLoss')),
    train_cfg=None,
    test_cfg=None)

dataset_type = 'PoseDataset'
ann_file_train = '/mnt/lustre/liguankai/data/ntu/nturgb+d_skeletons_60_3d/xsub/train.pkl'
ann_file_val = '/mnt/lustre/liguankai/data/ntu/nturgb+d_skeletons_60_3d/xsub/val.pkl'
train_pipeline = [
    # dict(type='PaddingWithLoop', clip_len=300),
    # dict(type='PoseDecode'),
    # dict(type='FormatGCNInput', input_format='NCTVM'),
    # dict(type='PoseNormalize'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    # dict(type='PaddingWithLoop', clip_len=300), # 新增frame_inds key
    # dict(type='PoseDecode'),  # frame_inds  np.float32
    # dict(type='FormatGCNInput', input_format='NCTVM'),
    # dict(type='PoseNormalize'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    # dict(type='PaddingWithLoop', clip_len=300),
    # dict(type='PoseDecode'),
    # dict(type='FormatGCNInput', input_format='NCTVM'),
    # dict(type='PoseNormalize'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=42,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix='',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix='',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix='',
        pipeline=test_pipeline))

# optimizer
optimizer = dict(
    type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001, nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[10, 50])
total_epochs = 80
checkpoint_config = dict(interval=3)
evaluation = dict(interval=3, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/stgcn_3d/'
load_from = None
resume_from = None
workflow = [('train', 1)]
