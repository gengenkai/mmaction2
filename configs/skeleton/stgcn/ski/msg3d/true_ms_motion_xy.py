model = dict(
    type='SkeletonGCN',
    backbone=dict(
        type='STGCN3',
        in_channels=3,
        # edge_importance_weighting=True,
        # adj_len=25,
        graph_cfg=dict(layout='ski', strategy='spatial')),
    cls_head=dict(
        type='STGCNHead',
        # num_classes=60,
        num_classes=30,
        in_channels=256,
        loss_cls=dict(type='CrossEntropyLoss'),
        num_person=1),
    train_cfg=None,
    test_cfg=None)

dataset_type = 'PoseDataset'
# ann_file_train = '/mnt/lustre/liguankai/data/ski/2500_422/no_padding/motion_xy/train_val.pkl'
ann_file_train = '/mnt/lustre/liguankai/data/ski/2500_422/no_padding/motion_xy/train.pkl'
ann_file_val = '/mnt/lustre/liguankai/data/ski/2500_422/no_padding/motion_xy/val.pkl'
# ann_file_val = '/mnt/lustre/liguankai/data/ski/2500_422/no_padding/motion_xy/test.pkl'
train_pipeline = [
    # dict(type='PaddingWithLoop', clip_len=2500),
    dict(type='UniformSampleFrames', clip_len=500),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput2', input_format='NCTVM', num_person=1),
    # dict(type='PoseNormalize'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    # dict(type='PaddingWithLoop', clip_len=2500), # 新增frame_inds key
    dict(type='UniformSampleFrames', clip_len=500),
    dict(type='PoseDecode'),  # frame_inds  np.float32
    dict(type='FormatGCNInput2', input_format='NCTVM', num_person=1),
    # dict(type='PoseNormalize'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    # dict(type='PaddingWithLoop', clip_len=2500),
    dict(type='UniformSampleFrames', clip_len=500),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput2', input_format='NCTVM',num_person=1),
    # dict(type='PoseNormalize'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=19,
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
lr_config = dict(policy='step', step=[30, 40])
# lr_config = dict(policy='step', step=[20, 50], by_epoch=True, warmup_iters=10)
# lr_config = dict(policy='CosineAnnealing', by_epoch=True, warmup_iters=10, min_lr=0) # modify lr
total_epochs = 80
checkpoint_config = dict(interval=3)
evaluation = dict(interval=3, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

# runtime settings
dist_params = dict(backend='nccl', port='1112')
log_level = 'INFO'
work_dir = './work_dirs/stgcn_3d/'
load_from = None
resume_from = None
workflow = [('train', 1)]
