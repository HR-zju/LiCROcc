work_dir = ''
_base_ = [
    '../_base_/default_runtime.py'
]
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

_sweeps_num_ = 10
_temporal_ = []

lims = [[-51.2, 51.2], [-51.2, 51.2], [-5, 3.0]]
sizes = [512, 512, 40]
grid_meters = [0.2, 0.2, 0.2]
nbr_classes = 17
phase = 'trainval'

ss_class_freq = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
sc_class_freq = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

grid_config = {
    'x': [-51.2, 51.2, 0.4],
    'y': [-51.2, 51.2, 0.4],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 45.0, 0.5],
}
numC_Trans = 64

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-0, 0),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}
# 👋

sync_bn = True
model = dict(
   type='SSC_RS',
   Distill_1 = True,
   Distill_2 = True,
   ratio_distill = 10,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=4,
        # frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch',
        pretrained='torchvision://resnet50',
        # frozen = True
    ),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0],
        temporal_adapter=True),
    img_view_transformer=dict(
        type='LSSViewTransformerBEVDepth',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        out_channels=numC_Trans,
        loss_depth_weight=1,
        depthnet_cfg=dict(use_dcn=False, aspp_mid_channels=96),
        downsample=16,
        temporal_adapter=True),
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_channels=[numC_Trans *2, numC_Trans * 4, numC_Trans * 8],
        temporal_adapter=True),
    # lidar
    pts_voxel_encoder=dict(
        type='PcPreprocessor',
        lims=lims,
        sizes=sizes, 
        grid_meters=grid_meters, 
        init_size=sizes[-1],
        frozen=True
    ),
    pts_backbone=dict(
        type='SemanticBranch',
        sizes=sizes,
        nbr_class=nbr_classes-1, 
        init_size=sizes[-1], 
        class_frequencies=ss_class_freq, 
        phase='test',
        frozen=True,
        ),
    pts_middle_encoder=dict(
        type='CompletionBranch',
        init_size=sizes[-1],
        nbr_class=nbr_classes,
        phase='test',
        frozen = True),
    pts_bbox_head=dict(
        type='BEVUNet',
        n_class=nbr_classes*sizes[-1],
        n_height=sizes[-1], 
        class_frequences=sc_class_freq,
        frozen = True,
        ),
    # radar
    radar_voxel_encoder=dict(
        type='PcPreprocessor',
        lims=lims,
        sizes=sizes, 
        grid_meters=grid_meters, 
        init_size=sizes[-1],
        frozen=False,
        pc_dim=13
    ),
    radar_backbone=dict(
        type='SemanticBranch',
        sizes=sizes,
        nbr_class=nbr_classes-1, 
        init_size=sizes[-1], 
        class_frequencies=ss_class_freq, 
        phase=phase,
        frozen=False
        ),
    radar_middle_encoder=dict(
        type='CompletionBranch',
        init_size=sizes[-1],
        nbr_class=nbr_classes,
        phase=phase,
        frozen = False),
    radar_bbox_head=dict(
        type='BEVUNet',
        n_class=nbr_classes*sizes[-1],
        n_height=sizes[-1], 
        class_frequences=sc_class_freq,
        use_cam=[False,False,False],
        use_Distill_2=True
        ),
    train_cfg=dict(pts=dict(
        sizes=sizes,
        grid_meters=grid_meters,
        lims=lims),
        sizes=sizes,
        ),
    test_cfg=dict(pts=dict(
        sizes=sizes,
        grid_meters=grid_meters,
        lims=lims),
        sizes=sizes)
    
)

load_from = 'pre_ckpt/merged_model_distill.pth'


dataset_type = 'nuScenesDataset'
data_root = './data/nuscenes/'
occ_root = './data/nuScenes-Occupancy'
file_client_args = dict(backend='disk')

data = dict(
   samples_per_gpu=4,
   workers_per_gpu=16,
   train=dict(
       type=dataset_type,
       split = "train",
       test_mode=False,
       data_root=data_root,
       occ_root=occ_root,
       lims=lims,
       sizes=sizes,
       temporal = _temporal_,
       sweeps_num=_sweeps_num_,
       augmentation=True,
       shuffle_index=True,
       data_config=data_config,
       grid_config=grid_config,
       ),
   val=dict(
       type=dataset_type,
       split = "val",
       test_mode=True,
       data_root=data_root,
       occ_root=occ_root,
       lims=lims,
       sizes=sizes,
       temporal = _temporal_,
       sweeps_num=_sweeps_num_,
       augmentation=False,
       shuffle_index=False,
       data_config=data_config,
       grid_config=grid_config,
       ),
   test=dict(
        type=dataset_type,
        split = "val",
        test_mode=True,
        data_root=data_root,
        occ_root=occ_root,
        lims=lims,
        sizes=sizes,
        temporal = _temporal_,
        sweeps_num=_sweeps_num_,
        augmentation=False,
        data_config=data_config,
        grid_config=grid_config,
        shuffle_index=False),
   shuffler_sampler=dict(type='DistributedGroupSampler'),
   nonshuffler_sampler=dict(type='DistributedSampler')
)
optimizer = dict(
   type='AdamW',
   lr=2e-4,
   weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
   policy='CosineAnnealing',
   warmup='linear',
   warmup_iters=500,
   warmup_ratio=1.0 / 3,
   min_lr_ratio=1e-3)
total_epochs = 30
evaluation = dict(interval=1)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
log_config = dict(
   interval=50,
   hooks=[
       dict(type='TextLoggerHook'),
       dict(type='TensorboardLoggerHook')
   ])

# checkpoint_config = None
checkpoint_config = dict(interval=1)
