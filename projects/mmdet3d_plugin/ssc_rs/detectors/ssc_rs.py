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
from thop import profile  
from mmdet3d.models.builder import build_backbone, build_neck, build_head, build_voxel_encoder, build_middle_encoder

def draw_feat(feats, type='img'):
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
                            )
        self.use_image = True if img_backbone!=None else False
        self.use_lidar = True if pts_voxel_encoder!=None else False
        self.use_radar = True if radar_voxel_encoder!=None else False
        self.use_image_for_distill = True if img_bev_encoder_backbone_distill!=None else False
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
            pass

        # distill model
        if img_backbone_distill:
            self.img_backbone_distill = build_backbone(img_backbone_distill)
            self.img_neck_distill = build_neck(img_neck_distill)
            self.img_view_transformer_distill = build_neck(img_view_transformer_distill)
            self.img_bev_encoder_backbone_distill = build_backbone(img_bev_encoder_backbone_distill)


        if radar_voxel_encoder:
            self.radar_voxel_encoder = build_voxel_encoder(
                radar_voxel_encoder)
            self.radar_middle_encoder = build_middle_encoder(
                radar_middle_encoder)
            self.radar_backbone = build_backbone(radar_backbone)
            radar_train_cfg = train_cfg.pts if train_cfg else None
            radar_bbox_head.update(train_cfg=radar_train_cfg)
            radar_test_cfg = test_cfg.pts if test_cfg else None
            radar_bbox_head.update(test_cfg=radar_test_cfg)
            self.radar_bbox_head = build_head(radar_bbox_head)
        if occ_head:
            self.occ_head = build_head(occ_head)
        if img_view_transformer: # 1
            self.img_view_transformer = build_neck(img_view_transformer)
        if img_bev_encoder_backbone: #1
            self.img_bev_encoder_backbone = build_backbone(img_bev_encoder_backbone)
        if img_bev_encoder_neck: #1
            self.img_bev_encoder_neck = build_neck(img_bev_encoder_neck)
        if occ_head:
            self.occ_head = build_head(occ_head)
            



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
                indicator.append(pc_i.size(0) + indicator[-1]) 
            pc = torch.cat(pc_ibatch, dim=0) 
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
        x, bev_t_list, _ = self.pts_bbox_head(inputs, ss_out_dict['mss_bev_dense'], sc_out_dict['mss_bev_dense'], img_feats)
        
        return x, ss_out_dict, sc_out_dict, bev_t_list

    def radar_step(self, points, img_metas, img_feats):
        batch_size = len(points)
        with torch.no_grad():
            indicator = [0]
            pc_ibatch = []
            for i in range(batch_size):
                pc_i = points[i]
                pc_ibatch.append(pc_i)
                indicator.append(pc_i.size(0) + indicator[-1]) 
            pc = torch.cat(pc_ibatch, dim=0) 
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
        x, bev_s_list, bev_s_residue_list = self.radar_bbox_head(inputs, ss_out_dict['mss_bev_dense'], sc_out_dict['mss_bev_dense'], img_feats)

        return x, ss_out_dict, sc_out_dict, bev_s_list, bev_s_residue_list

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
        if self.with_img_neck or True:

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
        x = self.img_bev_encoder_backbone(x) 

        if self.image_occ_loss:
            x_neck = self.img_bev_encoder_neck(x)
            if type(x_neck) in [list, tuple]:
                x_neck = x_neck[0]
        else:
            x_neck=None
        return x, x_neck

    @force_fp32()
    def bev_encoder_distill(self, x):
        """
        Args:
            x: (B, C, Dy, Dx)
        Returns:
            x: (B, C', 2*Dy, 2*Dx)
        """
        x = self.img_bev_encoder_backbone_distill(x) 
        if self.image_occ_loss:
            x_neck = self.img_bev_encoder_neck_distill(x)
            if type(x_neck) in [list, tuple]:
                x_neck = x_neck[0]
        else:
            x_neck=None
        # return x
        return x, x_neck


    def augmentation_random_flip(self, data, flip_type):  

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

        if img_inputs['flip_type'][0]!=-1:
            x =  self.augmentation_random_flip(x, img_inputs['flip_type'])
        x, x_neck = self.bev_encoder(x)

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
        

        x, _ = self.image_encoder_distill(imgs)   
        
        mlp_input = self.img_view_transformer_distill.get_mlp_input(
            sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda).float() 
 
        x, depth = self.img_view_transformer_distill([x, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda, mlp_input])
        

        if img_inputs['flip_type'][0]!=-1:
            x =  self.augmentation_random_flip(x, img_inputs['flip_type'])
        x, x_neck = self.bev_encoder_distill(x)

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
            
            nonzeros = target[i] != 0
            valid_mask = target[i] != 255
            valid = (valid_mask * nonzeros)
           
            bev_s_i = bev_s[i][valid]
            bev_t_i = bev_t[i][valid]
            if if_scene_useful[i]['if_scene_useful']:
                loss += nn.KLDivLoss(reduction="mean")(bev_s_i.unsqueeze(0), bev_t_i.unsqueeze(0))
            else:
                loss += nn.KLDivLoss(reduction="mean")(bev_s_i.unsqueeze(0), bev_t_i.unsqueeze(0))*1e-5   
        loss = loss / float(target.size(0))
        return loss * ratio

    def calculate_cosine_similarity(self, bev_radar_s, bev_fuser_t):

        assert bev_radar_s.shape == bev_fuser_t.shape, "输入特征的形状必须相同"
        B, C, H, W = bev_radar_s.shape

        bev_radar_s_flat = bev_radar_s.reshape(B, C, -1)
        bev_fuser_t_flat = bev_fuser_t.reshape(B, C, -1)


        bev_radar_s_norm = F.normalize(bev_radar_s_flat, p=2, dim=1)
        bev_fuser_t_norm = F.normalize(bev_fuser_t_flat, p=2, dim=1)

        cosine_similarity_flat = torch.bmm(bev_radar_s_norm.permute(0, 2, 1), bev_fuser_t_norm)

        return cosine_similarity_flat

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
        if self.use_image:
            img_feats, occ_bev_feature, depth = self.extract_img_feat(img_inputs)
            if self.Distill_1 or self.Distill_2 or self.Distill_3:
                if self.use_image_for_distill:
                    img_feats_distill, occ_bev_feature_distill, depth_distill = self.extract_img_feat_distill(img_inputs)
                
                    depth_distill_loss = self.img_view_transformer_distill.get_depth_loss(img_inputs['gt_depth'], depth_distill)
                else:
                    img_feats_distill = [None, None, None]
                    depth_distill_loss = None
                depth_loss = None
            else:
                depth_loss = self.img_view_transformer.get_depth_loss(img_inputs['gt_depth'], depth)
        else:
            img_feats = [None, None, None]
            img_feats_distill = [None, None, None]
            depth_loss = None 
        
       
        if self.use_lidar:
            bev_t, ss_out_dict, sc_out_dict, bev_t_list = self.step(points, img_metas, img_feats)
            if not self.use_radar:
                loss_seg_dict = self.pts_backbone.losses(ss_out_dict, img_metas)
                loss_com_dict = self.pts_middle_encoder.losses(sc_out_dict, img_metas)
                loss_ssc = self.pts_bbox_head.losses(bev_t, target)

        if self.use_radar:
            if self.use_lidar:
                if self.use_image_for_distill:
                    bev_s, ss_out_dict, sc_out_dict, bev_s_list, bev_s_residue_list = self.radar_step(radar_pc, img_metas, img_feats_distill)
                else:
                    bev_s, ss_out_dict, sc_out_dict, bev_s_list, bev_s_residue_list = self.radar_step(radar_pc, img_metas, [None, None, None])

            else:
                bev_s, ss_out_dict, sc_out_dict, bev_s_list, bev_s_residue_list = self.radar_step(radar_pc, img_metas, img_feats)

            loss_seg_dict = self.radar_backbone.losses(ss_out_dict, img_metas)
            loss_com_dict = self.radar_middle_encoder.losses(sc_out_dict, img_metas)
            loss_ssc = self.radar_bbox_head.losses(bev_s, target)


        if self.Distill_2:
            Distill_2_loss = 0
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
                    source_features = bev_s_residue_list[list_i][batch_i][:, mask_list[list_i]]
                    target_features = bev_t_list[list_i][batch_i][:, mask_list[list_i]]
                    cos_sim = cosine_similarity(source_features, target_features,dim=0)
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
        else:
            Distill_2_loss=None
            

        # 3stage distill
        if self.Distill_3:
            Distill_3_loss = 0.
            resize_shape = bev_s_list[-1].shape[-2:] 
            for i in range(len(bev_s_list)-1):
                i = i+1
                feature_target = bev_t_list[i].detach()
                feature_pred = bev_s_list[i]

                B, C, H, W = feature_pred.shape
                feature_pred_down = F.interpolate(feature_pred, size=resize_shape, mode="bilinear")
                feature_target_down = F.interpolate(feature_target, size=resize_shape, mode="bilinear")

                cosine_s = self.calculate_cosine_similarity(feature_pred_down, feature_pred_down)
                cosine_t = self.calculate_cosine_similarity(feature_target_down, feature_target_down)

                distill_3_loss = F.l1_loss(cosine_s, cosine_t, reduction='mean') / B
                Distill_3_loss += distill_3_loss
                
            Distill_3_loss *= self.ratio_distill*4

        
        if self.Distill_1: # kl
            # breakpoint()pip
            B,C,H,W,Z = bev_s.size()
            bev_s_softmax = F.log_softmax(bev_s, dim=1)  
            bev_t_softmax = F.softmax(bev_t, dim=1)

            bev_s_reshaped = bev_s_softmax.permute(0,2,3,4,1).reshape(B, H*W*Z, C) 
            bev_t_reshaped = bev_t_softmax.permute(0,2,3,4,1).reshape(B, H*W*Z, C) 
            target_reshaped = target.reshape(B,H*W*Z) 
  
            distill_loss_1 = self.Distill_1_loss(bev_s_reshaped, bev_t_reshaped, target_reshaped, self.ratio_distill, img_metas)

        losses = {'loss_semantic_scene': loss_ssc, 'loss_semantic_seg': sum(loss_seg_dict.values()), 'loss_scene_completion': sum(loss_com_dict.values()), }
        
        if depth_loss!=None:
            losses.update({'depth_loss': depth_loss})
        if self.use_image_for_distill:
            losses.update({'depth_distill_loss': depth_distill_loss})
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
                      **kwargs
                      ):
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
        if self.use_image:
            img_feats, occ_bev_feature, depth = self.extract_img_feat(img_inputs)
        else:
            img_feats = [None, None, None]
        if self.use_image_for_distill:
            img_feats_distill, occ_bev_feature_distill, depth_distill = self.extract_img_feat_distill(img_inputs)


        # if self.use_lidar:
        #     outs, _, _,_ = self.step(points, img_metas, img_feats)
        
        if self.use_radar:
            if self.use_lidar:
                if self.use_image_for_distill: # img_feats_distill
                    outs, _, _, _, _= self.radar_step(radar_pc, img_metas, img_feats_distill)
                    # breakpoint()
                else:
                    outs, _, _, _, _= self.radar_step(radar_pc, img_metas, [None,None,None])
            else:
                outs, _, _,_,_ = self.radar_step(radar_pc, img_metas, img_feats)
        elif self.use_lidar:
            outs, _, _,_ = self.step(points, img_metas, img_feats)

        result = dict()
        result['output_voxels'] = outs
        result['target_voxels'] = target

        return result
