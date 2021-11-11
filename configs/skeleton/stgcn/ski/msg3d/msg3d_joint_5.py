model = dict(
    type='SkeletonGCN',
    # backbone=dict(
    #     type='STGCN2',
    #     in_channels=3,
    #     edge_importance_weighting=True,
    #     adj_len=25,
    #     graph_cfg=dict(layout='ntu-rgb+d', strategy='spatial')),
    backbone=dict(
        type='MSG3D',
        in_channels=3,
        num_point=25,
        num_person=1,
        # num_gcn_scales=13,
        # num_g3d_scales=6,
        num_gcn_scales=5,
        num_g3d_scales=3,
        data_type='ski',
    ),
    cls_head=dict(
        type='STGCNHead',
        # num_classes=60,
        num_classes=30,
        in_channels=384,
        loss_cls=dict(type='CrossEntropyLoss'),
        num_person=1),
    train_cfg=None,
    test_cfg=None)

dataset_type = 'PoseDataset'
# ann_file_train = '/mnt/lustre/liguankai/data/ski/2500_422/padding_sub/train_val.pkl'
ann_file_train = '/mnt/lustre/liguankai/data/ski/2500_422/no_padding/train.pkl'
ann_file_val = '/mnt/lustre/liguankai/data/ski/2500_422/no_padding/val.pkl'
# ann_file_val = '/mnt/lustre/liguankai/data/ski/2500_422/padding_sub/test.pkl'
train_pipeline = [
    # dict(type='PaddingWithLoop', clip_len=1000),
    dict(type='UniformSampleFrames', clip_len=500),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput2', input_format='NCTVM', num_person=1),
    # dict(type='PoseNormalize'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    # dict(type='PaddingWithLoop', clip_len=1000), # 新增frame_inds key
    dict(type='UniformSampleFrames', clip_len=500),
    dict(type='PoseDecode'),  # frame_inds  np.float32
    dict(type='FormatGCNInput2', input_format='NCTVM', num_person=1),
    # dict(type='PoseNormalize'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    # dict(type='PaddingWithLoop', clip_len=1000),
    dict(type='UniformSampleFrames', clip_len=500),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput2', input_format='NCTVM',num_person=1),
    # dict(type='PoseNormalize'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=8,
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
# lr_config = dict(policy='step', step=[20, 40])  # modify lr
# lr_config = dict(policy='step', step=[20, 50], by_epoch=True, warmup_iters=10)
# lr_config = dict(policy='CosineAnnealing', by_epoch=True, warmup_iters=10, min_lr=0)
total_epochs = 70
checkpoint_config = dict(interval=3)
evaluation = dict(interval=3, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

# runtime settings
dist_params = dict(backend='nccl',port='1109')
log_level = 'INFO'
work_dir = './work_dirs/stgcn_3d/'
load_from = None
resume_from = None
workflow = [('train', 1)]
