from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmcv.runner import force_fp32
from PIL import Image  

import torch
import torch.nn.functional as F
import numpy as np
import time
import torch.nn as nn
from torch.nn.functional import cosine_similarity
from skimage.measure import block_reduce




def draw_feat(feats, type='img'):
    # 只显示第一个batch
    # breakpoint()
    mean_along_128 = [tensor.mean(dim=1)[0] for tensor in feats] 
    for idx, feat in enumerate(mean_along_128):
        image_tensor = feat
        image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())  
        image_array = image_tensor.detach().cpu().numpy()  
        image = Image.fromarray(np.uint8(image_array * 255))  
        image.save(f"{type}_{str(idx)}.png")  


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss #+ torch.log(1 + self.params[i] ** 2)
        return loss_sum
    

@DETECTORS.register_module()
class SSC_RS(MVXTwoStageDetector):
    def __init__(self,
                use_grid_mask=False,
                pts_voxel_layer=None,
                pts_voxel_encoder=None,
                pts_middle_encoder=None,
                pts_fusion_layer=None,
                img_backbone=None,
                pts_backbone=None,
                img_neck=None,
                pts_neck=None,
                pts_bbox_head=None,
                img_roi_head=None,
                img_rpn_head=None,
                train_cfg=None,
                test_cfg=None,
                pretrained=None,
                img_view_transformer=None,
                img_bev_encoder_backbone=None,
                img_bev_encoder_neck=None,
                use_radar=False,
                radar_voxel_encoder = None,
                radar_backbone=None,
                radar_middle_encoder = None,
                radar_bbox_head = None,
                occ_head =None,
                Distill_1 = False,
                Distill_2 = False,
                Distill_3 = False,
                Distill_3_mask = True,
                ratio_distill = 2,
                img_backbone_distill=None,
                img_neck_distill=None,
                img_view_transformer_distill=None,
                img_bev_encoder_backbone_distill = None,

                ):

        super(SSC_RS,
            self).__init__(pts_voxel_layer, pts_voxel_encoder,
                            pts_middle_encoder, pts_fusion_layer,
                            img_backbone, pts_backbone, img_neck, pts_neck,
                            pts_bbox_head, img_roi_head, img_rpn_head,
                            train_cfg, test_cfg, pretrained, 
                            img_view_transformer = img_view_transformer,
                            img_bev_encoder_backbone = img_bev_encoder_backbone, 
                            img_bev_encoder_neck=img_bev_encoder_neck,
                            radar_voxel_encoder = radar_voxel_encoder,
                            radar_backbone=radar_backbone,
                            radar_middle_encoder = radar_middle_encoder,
                            radar_bbox_head = radar_bbox_head,
                            occ_head = occ_head,
                            # distll
                            img_backbone_distill=img_backbone_distill,
                            img_neck_distill=img_neck_distill,
                            img_view_transformer_distill=img_view_transformer_distill,
                            img_bev_encoder_backbone_distill = img_bev_encoder_backbone_distill,
                            )
        self.use_image = True if img_backbone!=None else False
        self.use_lidar = True if pts_voxel_encoder!=None else False
        self.use_radar = True if radar_voxel_encoder!=None else False
        self.image_occ_loss= True if occ_head!=None else False
        self.Distill_1 = Distill_1
        self.Distill_2 = Distill_2
        self.Distill_3 = Distill_3
        self.Distill_3_mask = Distill_3_mask
        self.ratio_distill = ratio_distill
        self.distill_len = 0#self.Distill_1 + self.Distill_2 + self.Distill_3
        if self.distill_len >=1:
            self.awl = AutomaticWeightedLoss(self.distill_len)
        if self.Distill_3:
            channels = 512
            self.feat_conv = nn.Conv2d(channels, channels, 1)




    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            # breakpoint()
            return self.forward_train(**kwargs)

        else:
            return self.forward_test(**kwargs)

    def step(self, points, img_metas, img_feats):
        batch_size = len(points)
        with torch.no_grad():
            indicator = [0]
            pc_ibatch = []
            for i in range(batch_size):
                pc_i = points[i]
                pc_ibatch.append(pc_i)
                indicator.append(pc_i.size(0) + indicator[-1]) # 记录每个batch 开头的点数
            pc = torch.cat(pc_ibatch, dim=0) # torch.Size([1424022, 4])
        # breakpoint()
        vw_feature, coord_ind, full_coord, info = self.pts_voxel_encoder(pc, indicator)  # N, C; B, C, W, H, D       
        ss_out_dict = self.pts_backbone(vw_feature, coord_ind, full_coord, info)  # B, C, D, H, W

        occupancy = []
        for x in img_metas:
            occupancy.append(x['occupancy'])
        occupancy = np.stack(occupancy)
        occupancy = torch.from_numpy(occupancy).to(pc.device).permute(0, 3, 2, 1) # B, D, H, W
        sc_out_dict = self.pts_middle_encoder(occupancy.unsqueeze(1))

        coord = torch.cat([coord_ind[:, 0].reshape(-1, 1), torch.flip(coord_ind, dims=[1])[:, :3]], dim=1)
        sizes = self.train_cfg.sizes if self.train_cfg is not None else self.test_cfg.sizes
        bev_dense = self.pts_backbone.bev_projection(vw_feature, coord, np.array(sizes, np.int32)[::-1], batch_size) # B, C, H, W
        inputs = torch.cat([occupancy, bev_dense], dim=1)  # B, C, H, W
        x, bev_t_list, _, pc_t_list = self.pts_bbox_head(inputs, ss_out_dict['mss_bev_dense'], sc_out_dict['mss_bev_dense'], img_feats)
        
        return x, ss_out_dict, sc_out_dict, bev_t_list, pc_t_list

    def radar_step(self, points, img_metas, img_feats):
        batch_size = len(points)
        with torch.no_grad():
            indicator = [0]
            pc_ibatch = []
            for i in range(batch_size):
                pc_i = points[i]
                pc_ibatch.append(pc_i)
                indicator.append(pc_i.size(0) + indicator[-1]) # 记录每个batch 开头的点数
            pc = torch.cat(pc_ibatch, dim=0) # torch.Size([1424022, 4])
        # breakpoint()
        vw_feature, coord_ind, full_coord, info = self.radar_voxel_encoder(pc, indicator)  # N, C; B, C, W, H, D       
        ss_out_dict = self.radar_backbone(vw_feature, coord_ind, full_coord, info)  # B, C, D, H, W

        occupancy = []
        for x in img_metas:
            occupancy.append(x['radar_occ'])
        occupancy = np.stack(occupancy)
        occupancy = torch.from_numpy(occupancy).to(pc.device).permute(0, 3, 2, 1) # B, D, H, W
        sc_out_dict = self.radar_middle_encoder(occupancy.unsqueeze(1))

        coord = torch.cat([coord_ind[:, 0].reshape(-1, 1), torch.flip(coord_ind, dims=[1])[:, :3]], dim=1)
        sizes = self.train_cfg.sizes if self.train_cfg is not None else self.test_cfg.sizes
        bev_dense = self.radar_backbone.bev_projection(vw_feature, coord, np.array(sizes, np.int32)[::-1], batch_size) # B, C, H, W
        inputs = torch.cat([occupancy, bev_dense], dim=1)  # B, C, H, W
        # x, bev_s_list = self.radar_bbox_head(inputs, ss_out_dict['mss_bev_dense'], sc_out_dict['mss_bev_dense'], img_feats)
        x, bev_s_list, bev_s_residue_list, pc_s_list = self.radar_bbox_head(inputs, ss_out_dict['mss_bev_dense'], sc_out_dict['mss_bev_dense'], img_feats)

        return x, ss_out_dict, sc_out_dict, bev_s_list, bev_s_residue_list, pc_s_list

    def prepare_inputs(self, inputs):
        return inputs['imgs'], inputs['sensor2egos'], None, inputs['intrins'], inputs['post_rots'], inputs['post_trans'], None
    
    def image_encoder(self, img, stereo=False):
        """
        Args:
            img: (B, N, 3, H, W)
            stereo: bool
        Returns:
            x: (B, N, C, fH, fW)
            stereo_feat: (B*N, C_stereo, fH_stereo, fW_stereo) / None
        """
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)
        stereo_feat = None
        if stereo:
            stereo_feat = x[0]
            x = x[1:]
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x, stereo_feat

    def image_encoder_distill(self, img, stereo=False):
        """
        Args:
            img: (B, N, 3, H, W)
            stereo: bool
        Returns:
            x: (B, N, C, fH, fW)
            stereo_feat: (B*N, C_stereo, fH_stereo, fW_stereo) / None
        """
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone_distill(imgs)
        stereo_feat = None
        if stereo:
            stereo_feat = x[0]
            x = x[1:]
        if self.with_img_neck:
            x = self.img_neck_distill(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x, stereo_feat

    @force_fp32()
    def bev_encoder(self, x):
        """
        Args:
            x: (B, C, Dy, Dx)
        Returns:
            x: (B, C', 2*Dy, 2*Dx)
        """
        # breakpoint()
        x = self.img_bev_encoder_backbone(x) # torch.Size([1, 64, 200, 200])
        
        # (Pdb) x[0].size()
        # torch.Size([1, 128, 100, 100])
        # (Pdb) x[1].size()
        # torch.Size([1, 256, 50, 50])
        # (Pdb) x[2].size()
        # torch.Size([1, 512, 25, 25])
        # breakpoint()
        if self.image_occ_loss:
            x_neck = self.img_bev_encoder_neck(x)
            if type(x_neck) in [list, tuple]:
                x_neck = x_neck[0]
        else:
            x_neck=None
        # return x
        return x, x_neck

    @force_fp32()
    def bev_encoder_distill(self, x):
        """
        Args:
            x: (B, C, Dy, Dx)
        Returns:
            x: (B, C', 2*Dy, 2*Dx)
        """
        # breakpoint()
        x = self.img_bev_encoder_backbone_distill(x) # torch.Size([1, 64, 200, 200])
        
        # (Pdb) x[0].size()
        # torch.Size([1, 128, 100, 100])
        # (Pdb) x[1].size()
        # torch.Size([1, 256, 50, 50])
        # (Pdb) x[2].size()
        # torch.Size([1, 512, 25, 25])
        # breakpoint()
        if self.image_occ_loss:
            x_neck = self.img_bev_encoder_neck_distill(x)
            if type(x_neck) in [list, tuple]:
                x_neck = x_neck[0]
        else:
            x_neck=None
        # return x
        return x, x_neck


    # def augmentation_random_flip(self, data, flip_type):
    #     for j in range(len(data)):
    #         for i in range(len(flip_type)):
    #             if flip_type[i]==1:
    #                 data[j][i] = torch.flip(data[j][i], dims=[-1])

    #             elif flip_type[i]==2:
    #                 data[j][i] = torch.flip(data[j][i], dims=[-2])
    #             elif flip_type[i]==3:
    #                 data[j][i] = torch.flip(torch.flip(data[j][i], dims=[-1]), dims=[-2])
    #     return data
    # def augmentation_random_flip(self, data, flip_type):  
    #     augmented_data = [torch.clone(item) for item in data]  # 创建数据副本  ]
    #     for j in range(len(augmented_data)):  
    #         for i in range(len(flip_type)):  
    #             if flip_type[i] == 1:  
    #                 augmented_data[j][i] = torch.flip(augmented_data[j][i], dims=[-1]).clone() 
    #             elif flip_type[i] == 2:  
    #                 augmented_data[j][i] = torch.flip(augmented_data[j][i], dims=[-2]).clone()   
    #             elif flip_type[i] == 3:  
    #                 augmented_data[j][i] = torch.flip(torch.flip(augmented_data[j][i], dims=[-1]), dims=[-2]).clone()  
    #     return augmented_data  
    def augmentation_random_flip(self, data, flip_type):  
        # augmented_data = [torch.clone(item) for item in data]  # 创建数据副本  ]
        # for j in range(len(data)):  
        augmented_data = data.clone()
        for i in range(len(flip_type)):  
            if flip_type[i] == 1:  
                augmented_data[i] = torch.flip(augmented_data[i], dims=[-1]).clone() 
            elif flip_type[i] == 2:  
                augmented_data[i] = torch.flip(augmented_data[i], dims=[-2]).clone()   
            elif flip_type[i] == 3:  
                augmented_data[i] = torch.flip(torch.flip(augmented_data[i], dims=[-1]), dims=[-2]).clone()  
        return augmented_data  

        
    def extract_img_feat(self, img_inputs):
        """ Extract features of images.
        img_inputs:
            imgs:  (B, N_views, 3, H, W)
            sensor2egos: (B, N_views, 4, 4)
            ego2globals: (B, N_views, 4, 4)
            intrins:     (B, N_views, 3, 3)
            post_rots:   (B, N_views, 3, 3)
            post_trans:  (B, N_views, 3)
            bda_rot:  (B, 3, 3)
        Returns:
            x: [(B, C', H', W'), ]
            depth: (B*N, D, fH, fW)
        """
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda = self.prepare_inputs(img_inputs)

        x, _ = self.image_encoder(imgs)    # x: (B, N, C, fH, fW) torch.Size([1, 6, 256, 16, 44])
        mlp_input = self.img_view_transformer.get_mlp_input(
            sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda).float()  # (B, N_views, 27)
        x, depth = self.img_view_transformer([x, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda, mlp_input])
        # x: (B, C, Dy, Dx)
        # depth: (B*N, D, fH, fW)
        # encode前filp
        if img_inputs['flip_type'][0]!=-1:
            x =  self.augmentation_random_flip(x, img_inputs['flip_type'])
        x, x_neck = self.bev_encoder(x)
        # breakpoint()
        # if img_inputs['flip_type'][0]!=-1:
        #     x = self.augmentation_random_flip(x, img_inputs['flip_type'])
        return x, x_neck, depth

    def extract_img_feat_distill(self, img_inputs):
        """ Extract features of images.
        img_inputs:
            imgs:  (B, N_views, 3, H, W)
            sensor2egos: (B, N_views, 4, 4)
            ego2globals: (B, N_views, 4, 4)
            intrins:     (B, N_views, 3, 3)
            post_rots:   (B, N_views, 3, 3)
            post_trans:  (B, N_views, 3)
            bda_rot:  (B, 3, 3)
        Returns:
            x: [(B, C', H', W'), ]
            depth: (B*N, D, fH, fW)
        """
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda = self.prepare_inputs(img_inputs)

        x, _ = self.image_encoder_distill(imgs)    # x: (B, N, C, fH, fW) torch.Size([1, 6, 256, 16, 44])
        mlp_input = self.img_view_transformer_distill.get_mlp_input(
            sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda).float()  # (B, N_views, 27)
        x, depth = self.img_view_transformer_distill([x, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda, mlp_input])
        # x: (B, C, Dy, Dx)
        # depth: (B*N, D, fH, fW)
        # encode前filp
        if img_inputs['flip_type'][0]!=-1:
            x =  self.augmentation_random_flip(x, img_inputs['flip_type'])
        x, x_neck = self.bev_encoder_distill(x)
        # breakpoint()
        # if img_inputs['flip_type'][0]!=-1:
        #     x = self.augmentation_random_flip(x, img_inputs['flip_type'])
        return x, x_neck, depth


    def forward_image_occ(self, img_feats, voxel_semantics, mask_camera):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:
        """
        outs = self.occ_head(img_feats)
        # assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        # breakpoint()
        loss_occ = self.occ_head.loss(
            outs,  # (B, Dx, Dy, Dz, n_cls)
            voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera,  # (B, Dx, Dy, Dz)
        )
        return loss_occ

    def Distill_1_loss(self, bev_s, bev_t, target, ratio, if_scene_useful):
        """
        KL divergence on nonzeros classes
        nonzeros =target target != 0
        """ 
        loss = 0
        for i in range(target.size(0)):
            # target = target.view(-1)
            
            nonzeros = target[i] != 0
            valid_mask = target[i] != 255
            valid = (valid_mask * nonzeros)
            # bev_s = bev_s.squeeze()[:,valid]
            # bev_t = bev_t.squeeze()[:,valid]
            bev_s_i = bev_s[i][valid]
            bev_t_i = bev_t[i][valid]
            # loss = nn.MSELoss()(bev_s.unsqueeze(0), bev_t.unsqueeze(0))
            if if_scene_useful[i]['if_scene_useful']:
                loss += nn.KLDivLoss(reduction="mean")(bev_s_i.unsqueeze(0), bev_t_i.unsqueeze(0))
            else:
                loss += nn.KLDivLoss(reduction="mean")(bev_s_i.unsqueeze(0), bev_t_i.unsqueeze(0))*1e-5   
        loss = loss / float(target.size(0))
        return loss * ratio

    def calculate_cosine_similarity(self, bev_radar_s, bev_fuser_t, mask):
        # bev_radar_s 和 bev_fuser_t 应当有相同的形状 (B, C, H, W)
        # breakpoint()
        # 验证形状是否一致
        assert bev_radar_s.shape == bev_fuser_t.shape, "输入特征的形状必须相同"

        # 扁平化特征的H和W维度，形状从 (B, C, H, W) 变为 (B, C, H*W)
        bev_radar_s_flat = bev_radar_s[:,mask]
        bev_fuser_t_flat = bev_fuser_t[:,mask]
        # breakpoint()

        # # 计算归一化后的特征，使特征在C维度上具有单位长度
        bev_radar_s_norm = F.normalize(bev_radar_s_flat, p=2, dim=0)
        bev_fuser_t_norm = F.normalize(bev_fuser_t_flat, p=2, dim=0)

        # # 使用批量矩阵乘法计算cosine相似度 (B, H*W, H*W)
        cosine_similarity_flat = torch.bmm(bev_radar_s_norm.unsqueeze(0).transpose(1, 2), bev_fuser_t_norm.unsqueeze(0))
        # cos_sim = cosine_similarity(bev_radar_s_flat, bev_fuser_t_flat, dim=0)


        # 重塑cosine相似度矩阵为 (B, H, W) 形状
        # cosine_similarity = cosine_similarity_flat.view(bev_radar_s.size(0), bev_radar_s.size(2)*bev_radar_s.size(3), bev_radar_s.size(2)*bev_radar_s.size(3))

        return cosine_similarity_flat.squeeze(0)

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                    img_metas=None,
                    points=None,
                    target=None,
                    img_inputs=None,
                    radar_pc=None,
                    scene_token=None):
        """Forward training function.
        Args:
            metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            points (torch.Tensor): Images of each sample with shape
                (batch, C, H, W). Defaults to None.
            target (torch.Tensor): ground-truth of semantic scene completion
                (batch, X_grids, Y_grids, Z_grids)
        Returns:
            dict: Losses of different branches.
        """
        # breakpoint()
        # forward_start_time = time.time()
        # breakpoint()
        if self.use_image:
            img_feats, occ_bev_feature, depth = self.extract_img_feat(img_inputs)
        # 计算depth_loss
    # breakpoint()
            if self.Distill_1 or self.Distill_2 or self.Distill_3:
                # img_feats_distill, occ_bev_feature_distill, depth_distill = self.extract_img_feat_distill(img_inputs)
                # depth_loss = self.img_view_transformer_distill.get_depth_loss(img_inputs['gt_depth'], depth_distill)
                img_feats_distill = [None, None, None]
                depth_loss = None
            else:
                depth_loss = self.img_view_transformer.get_depth_loss(img_inputs['gt_depth'], depth)
        else:
            img_feats = [None, None, None]
        #     img_feats_distill = [None, None, None]
            depth_loss = None 
        
        # 如果有图像分支单独监督
        if self.image_occ_loss:
            # breakpoint()
            loss_occ = self.forward_image_occ(occ_bev_feature, img_metas, 0)
            # pass

            # (Pdb) img_metas[0]['target_1_2'].shape
            # (256, 256, 20)
            # (Pdb) img_metas[0]['target_1_2'].max
            # <built-in method max of numpy.ndarray object at 0x7fd9dc87e8a0>
            # (Pdb) img_metas[0]['target_1_2'].max()
            # 255
            # (Pdb) img_metas[0]['target_1_2'].min()
            # 0
            # (Pdb) (img_metas[0]['target_1_2']==17).sum()
            # 0
            # (Pdb) (img_metas[0]['target_1_2']==16).sum()
            # 6481
            # (Pdb) (img_metas[0]['target_1_2']==15).sum()

  
        # -----------------------------------
        # breakpoint()
        # depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
        # depth_preds = self.img_view_transformer.get_depth_loss(depth)

        # save image
        # mean = [0.485, 0.456, 0.406]  
        # std = [0.229, 0.224, 0.225]  
        # denormalize = transforms.Normalize(mean=[-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]], std=[1/std[0], 1/std[1], 1/std[2]])  
        # tensor_image = img_inputs['imgs']
        # denormalized_tensor = denormalize(tensor_image)  

        # numpy_image = tensor_image.cpu().numpy()  
        # pil_image = Image.fromarray(numpy_image.transpose(1, 2, 0).astype('uint8'))  


        # breakpoint()
        if self.use_lidar:
            bev_t, ss_out_dict, sc_out_dict, bev_t_list, pc_t_list = self.step(points, img_metas, img_feats)
        # TODO image brach
        # 检查一下深度图是不是对的 可以检查模型是否加载正确？
        #  lidar bev特征 
            if not self.use_radar:
                loss_seg_dict = self.pts_backbone.losses(ss_out_dict, img_metas)
                loss_com_dict = self.pts_middle_encoder.losses(sc_out_dict, img_metas)
                loss_ssc = self.pts_bbox_head.losses(bev_t, target)
        # breakpoint()

        if self.use_radar:
            # bev_s, ss_out_dict, sc_out_dict, bev_s_list = self.radar_step(radar_pc, img_metas, [None, None, None])
            if self.use_lidar:
                bev_s, ss_out_dict, sc_out_dict, bev_s_list, bev_s_residue_list, pc_s_list = self.radar_step(radar_pc, img_metas, [None, None, None])
            else:
                bev_s, ss_out_dict, sc_out_dict, bev_s_list, bev_s_residue_list, pc_s_list = self.radar_step(radar_pc, img_metas, img_feats)


            # [0] torch.Size([4, 128, 256, 256])
            # [1] torch.Size([4, 256, 128, 128])
            # forward_end_time = time.time()
            
            loss_seg_dict = self.radar_backbone.losses(ss_out_dict, img_metas)
            loss_com_dict = self.radar_middle_encoder.losses(sc_out_dict, img_metas)
            loss_ssc = self.radar_bbox_head.losses(bev_s, target)
            # loss_end_time = time.time()
        # print('forward_time = {}'.format(forward_end_time-forward_start_time))
        # print('loss_time = {}'.format(loss_end_time-forward_end_time))
        if self.Distill_2:
            # draw_feat(bev_s_list,'stu')
            # draw_feat(bev_t_list,'tea')
            # d2_list=[]
            Distill_2_loss = 0
            # for i in range(len(bev_s_list)):
            #     # bev_radar_s = bev_s_list[i]
            #     B,C,H,W = bev_s_list[i].size()
                
            #     # breakpoint()
            #     d2_list.append(torch.sqrt(torch.sum((bev_s_list[i] - bev_t_list[i])**2, dim=(1))).unsqueeze(1))
            #     norms = torch.mean(torch.sqrt(torch.sum((bev_s_list[i] - bev_t_list[i])**2, dim=(1))), dim=(1,2))
            #     average_norm = torch.mean(norms)  
            #     Distill_2_loss += average_norm*0.2
            # draw_feat(d2_list,'dis')
            # breakpoint()
            
            # L2 diss
            # for batch_i in range(len(img_metas)):
            #     mask_1_2 = img_metas[batch_i]['target_1_2']
            #     mask_1_4 = img_metas[batch_i]['target_1_4']
            #     mask_1_8 = img_metas[batch_i]['target_1_8']
            #     mask_1_2[mask_1_2==255] = 0
            #     mask_1_4[mask_1_4==255] = 0
            #     mask_1_8[mask_1_8==255] = 0

            #     mask_list = [
            #         np.mean(mask_1_2, axis=-1)!=0,
            #         np.mean(mask_1_4, axis=-1)!=0,
            #         np.mean(mask_1_8, axis=-1)!=0,
            #     ]
            #     for list_i in range(len(bev_s_list)):
            #     # bev_radar_s = bev_s_list[i]
            #         C,H,W = bev_s_list[list_i][batch_i].size()
                    
            #         # breakpoint()
            #         # d2_list.append(torch.sqrt(torch.sum((bev_s_list[list_i][batch_i][:,mask_list[list_i]] - bev_t_list[list_i][batch_i][:,mask_list[list_i]])**2, dim=(0))).unsqueeze(1))
            #         norms = torch.sqrt(torch.sum((bev_s_list[list_i][batch_i][:,mask_list[list_i]] - bev_t_list[list_i][batch_i][:,mask_list[list_i]])**2, dim=(0)))
            #         average_norm = torch.mean(norms)  
            #         Distill_2_loss += average_norm *0.2
            # Distill_2_loss /= float(len(bev_s_list)*len(img_metas))
            # breakpoint()
            # cosine
            # 
            for batch_i in range(len(img_metas)):
                mask_1_2 = img_metas[batch_i]['target_1_2']
                mask_1_4 = img_metas[batch_i]['target_1_4']
                mask_1_8 = img_metas[batch_i]['target_1_8']
                mask_1_2[mask_1_2 == 255] = 0
                mask_1_4[mask_1_4 == 255] = 0
                mask_1_8[mask_1_8 == 255] = 0

                mask_list = [
                    np.mean(mask_1_2, axis=-1) != 0,
                    np.mean(mask_1_4, axis=-1) != 0,
                    np.mean(mask_1_8, axis=-1) != 0,
                ]

                for list_i in range(len(bev_s_residue_list)):
                    # Get the current batch's feature size
                    C, H, W = bev_s_residue_list[list_i][batch_i].size()
                    
                    # Select the masked elements from both source and target features
                    # breakpoint()
                    # bev_s_residue = self.distill_mlp[list_i](bev_s_list[list_i][batch_i])
                    # bev_s_list[list_i][batch_i] = (bev_s_residue+bev_s_list[list_i][batch_i]).clone()
                    source_features = bev_s_residue_list[list_i][batch_i][:, mask_list[list_i]]

                    target_features = bev_t_list[list_i][batch_i][:, mask_list[list_i]]
                    
                    # Calculate the cosine similarity for the selected elements
                    # Note: cosine_similarity expects inputs of shape (1, C) or (N, C), so we might need to unsqueeze dimensions
                    cos_sim = cosine_similarity(source_features, target_features,dim=0)
                    # breakpoint()
                    # Now obtain the cosine distance
                    cos_distance = 1 - cos_sim
                    
                    # Compute the mean of cosine distances
                    average_distance = torch.mean(cos_distance)
                    
                    # Update the Distill_2_loss with the weighted average cosine distance
                    if img_metas[batch_i]['if_scene_useful']:
                        Distill_2_loss += average_distance
                    else:
                        Distill_2_loss += average_distance * 1e-5


            # Normalize the Distill_2_loss by the number of list and meta elements
            Distill_2_loss /= float(len(bev_s_residue_list) * len(img_metas))
            Distill_2_loss *= 4.
            # breakpoint()
            



            # Distill_2_loss = 0.0  
            # for bev_radar_s, bev_fuser_t in zip(bev_s_list, bev_t_list):  
            #     diff =bev_radar_s - bev_fuser_t  # 计算特征之间的差异  
            #     # breakpoint()
            #     norms = torch.norm(diff.view(diff.size(0), -1), p=2, dim=1)  # 计算二范数  
            #     average_norm = torch.mean(norms / float(diff.size(2) * diff.size(3)))  # 计算平均二范数  
            #     Distill_2_loss += average_norm  # 累加到 Distill_2_loss 
            # Distill_2_loss *= self.ratio_distill
        else:
            Distill_2_loss=None
            

        if self.Distill_3:
            # bev_radar_s_relationship = bev_s_list[-1]
            # bev_fuser_t_relationship = bev_t_list[-1]
            Distill_3_loss = 0.
            bev_s_x5 = self.feat_conv(bev_s_list[-1])
            for batch_i in range(len(img_metas)):
                if self.Distill_3_mask:
                    mask_1_8 = img_metas[batch_i]['target_1_8']
                    mask_1_8[mask_1_8==255] = 0
                    mask = np.mean(mask_1_8, axis=-1)!=0
                    # breakpoint()
                    mask = block_reduce(mask, (2,2), np.max)

                else:
                    # breakpoint()
                    # mask = np.full((64, 64), True, dtype=bool)  
                    mask = np.full((32, 32), True, dtype=bool)  


                # mask_list = [
                #     np.mean(mask_1_8, axis=-1)!=0,
                # ]
                # breakpoint()
                # draw_feat(bev_s_list,'bev_s_list')
                # draw_feat(bev_t_list,'bev_t_list')
                # breakpoint() # b,c,h,w
                # mask_has_radar = np.mean(bev_s_x5.detach().cpu().numpy(), axis=1)!=0
                cosine_s = self.calculate_cosine_similarity(bev_s_x5[batch_i], bev_s_x5[batch_i], mask )
                cosine_t = self.calculate_cosine_similarity(bev_t_list[-1][batch_i], bev_t_list[-1][batch_i], mask)
                diff_abs = torch.abs(cosine_s - cosine_t)
                # breakpoint()

            # 计算H和W维度的累计一范数，结果形状为(B,)
                l1_norm = torch.sum(diff_abs)
                distill_3_loss = l1_norm/float(diff_abs.size(0)*diff_abs.size(0))
                if img_metas[batch_i]['if_scene_useful']:
                    Distill_3_loss += distill_3_loss
                else:
                    Distill_3_loss += distill_3_loss * 1e-5

            Distill_3_loss *= self.ratio_distill / len(img_metas) # /B
            # breakpoint()




        # tensor(0.1213, device='cuda:0', grad_fn=<AddBackward0>)
        # tensor(0.1177, device='cuda:0', grad_fn=<AddBackward0>)
        # breakpoint()
        # # 0
        # image_tensor = mean_along_128[0]
        # image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())  
        # image_array = image_tensor.detach().cpu().numpy()  
        # image = Image.fromarray(np.uint8(image_array * 255))  
        # image.save("img_0.png")  
        # # 1
        # image_tensor = mean_along_128[1]
        # image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())  
        # image_array = image_tensor.detach().cpu().numpy()  
        # image = Image.fromarray(np.uint8(image_array * 255))  
        # image.save("img_1.png")  
        # # 2
        # image_tensor = mean_along_128[2]
        # image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())  
        # image_array = image_tensor.detach().cpu().numpy()  
        # image = Image.fromarray(np.uint8(image_array * 255))  
        # image.save("img_2.png")  
        # -----------------------------------
        
        if self.Distill_1: # kl
            # breakpoint()
            B,C,H,W,Z = bev_s.size()
            bev_s_softmax = F.log_softmax(bev_s, dim=1)  
            bev_t_softmax = F.softmax(bev_t, dim=1)


            # 将bev_s和bev_t调整为二维张量  
            bev_s_reshaped = bev_s_softmax.permute(0,2,3,4,1).reshape(B, H*W*Z, C)  # 将除了batch size外的维度合并为一维  
            bev_t_reshaped = bev_t_softmax.permute(0,2,3,4,1).reshape(B, H*W*Z, C)  # 将除了batch size外的维度合并为一维
            target_reshaped = target.reshape(B,H*W*Z)  # 将除了batch size外的维度合并为一维  
  
            distill_loss_1 = self.Distill_1_loss(bev_s_reshaped, bev_t_reshaped, target_reshaped, self.ratio_distill, img_metas)
            # pass
        # bev_s = torch.Size([4, 17, 512, 512, 40])
        # breakpoint()



        losses = {'loss_semantic_scene': loss_ssc, 'loss_semantic_seg': sum(loss_seg_dict.values()), 'loss_scene_completion': sum(loss_com_dict.values()), }
        # losses = {'loss_semantic_scene': loss_ssc}
        # losses = {'loss_semantic_scene': loss_ssc, 'loss_semantic_seg': None, 'loss_scene_completion': None}
        if depth_loss!=None:
            losses.update({'depth_loss': depth_loss})
        if self.image_occ_loss:
            losses.update(loss_occ)

        if self.distill_len>=1:
            distill_loss = []
            if self.Distill_1:
                distill_loss.append(distill_loss_1)
            if Distill_2_loss != None:
                distill_loss.append(Distill_2_loss)
                # losses.update({'Distill_2_loss':Distill_2_loss})
            if self.Distill_3:
                distill_loss.append(Distill_3_loss)
            distill_loss_add = self.awl(distill_loss)

            losses.update({'Distill_loss':distill_loss_add})
                # losses.update({'Distill_3_loss':Distill_3_loss})
            # self.awl
        else:
            if self.Distill_1:
                losses.update({'distill_loss_1':distill_loss_1})
            if Distill_2_loss != None:
                losses.update({'Distill_2_loss':Distill_2_loss})
            if self.Distill_3:
                losses.update({'Distill_3_loss':Distill_3_loss})

        
        return losses

    def forward_test(self,
                    img_metas=None,
                    points=None,
                    target=None,
                    img_inputs = None,
                    radar_pc=None,
                    scene_token=None,
                      **kwargs):
        """Forward testing function.
        Args:
            metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            points (torch.Tensor): Images of each sample with shape
                (batch, C, H, W). Defaults to None.
            target (torch.Tensor): ground-truth of semantic scene completion
                (batch, X_grids, Y_grids, Z_grids)
        Returns:
            dict: Completion result.
        """
        # img_feats, depth = self.extract_img_feat(img_inputs)
        # breakpoint()
        if self.use_image:
            img_feats, occ_bev_feature, depth = self.extract_img_feat(img_inputs)
        else:
            img_feats = [None, None, None]

        if self.use_lidar:
            outs, _, _,_,_ = self.step(points, img_metas, img_feats)
        if self.use_radar:
            if self.use_lidar:
                outs, _, _,_,_,_ = self.radar_step(radar_pc, img_metas, [None,None,None])
            else:
                outs, _, _,_,_ ,_= self.radar_step(radar_pc, img_metas, img_feats)


        result = dict()
        result['output_voxels'] = outs
        result['target_voxels'] = target

        return result
