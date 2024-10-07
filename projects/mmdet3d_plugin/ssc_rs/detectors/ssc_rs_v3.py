from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

import torch
import torch.nn.functional as F
import numpy as np

from projects.mmdet3d_plugin.ssc_rs.modules.cspn import Affinity_Propagate

@DETECTORS.register_module()
class SSC_RS_V3(MVXTwoStageDetector):
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
                 target_scales=None,
                 warpping=True,
                 ):

        super(SSC_RS_V3,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)

        self.prev_info = {
            'prev_bev': None,
            'sequence_id': None,
            'prev_lidar2global': None
        }
        self.target_scales = target_scales
        self.warpping = warpping
        
        self.cspn = Affinity_Propagate(prop_time=1)
    
        for k, v in self.named_parameters():
            if 'guidance_conv' not in k:
                v.requires_grad = False
            else:
                v.requires_grad = True

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

    def step(self, points, occupancy, return_bev=False, prev_bev=None):
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

        occupancy = occupancy.permute(0, 3, 2, 1) # B, D, H, W
        sc_out_dict = self.pts_middle_encoder(occupancy.unsqueeze(1))

        coord = torch.cat([coord_ind[:, 0].reshape(-1, 1), torch.flip(coord_ind, dims=[1])[:, :3]], dim=1)
        sizes = self.train_cfg.pts.sizes if self.train_cfg is not None else self.test_cfg.pts.sizes
        bev_dense = self.pts_backbone.bev_projection(vw_feature, coord, np.array(sizes, np.int32)[::-1], batch_size) # B, C, H, W
        inputs = torch.cat([occupancy, bev_dense], dim=1)  # B, C, H, W
        x = self.pts_bbox_head(inputs, ss_out_dict['mss_bev_dense'], sc_out_dict['mss_bev_dense'], return_bev=return_bev, prev_bev=prev_bev)
        
        return x, ss_out_dict, sc_out_dict

    def feature_warpping(self, prev_bev, prev_lidar2global, lidar2global):
        curr2prev_ = torch.matmul(torch.inverse(prev_lidar2global), lidar2global).float()
        curr2prev = torch.zeros_like(curr2prev_[:, :3, :3])
        curr2prev[:, :2, :2] = curr2prev_[:, :2, :2]
        curr2prev[:, :2, 2] = curr2prev_[:, :2, 3]
        curr2prev[:, 2, 2] = 1
        lims = self.train_cfg.pts.lims if self.train_cfg is not None else self.test_cfg.pts.lims
        origin = [lims[0][0], lims[1][0]]
        origin = torch.from_numpy(np.array(origin, dtype=np.float32)).to(lidar2global.device)

        warpped_bev = tuple()
        target_scales_dict = {0:'1_2', 1:'1_4', 2:'1_8', 3:'1_16'}
        for i, x in enumerate(prev_bev):
            if target_scales_dict[i] in self.target_scales:
                shape = x[0].shape[-2:]
                resolution = (lims[0][1]-lims[0][0])/shape[-1]
                yv, xv = np.meshgrid(range(shape[0]), range(shape[1]), indexing='ij')
                grids = torch.from_numpy(np.concatenate([(xv.reshape(1,-1)+0.5), (yv.reshape(1,-1)+0.5)], axis=0)).float().T.to(x.device)
                points = grids * resolution + origin
                points = torch.cat([points, torch.ones_like(grids[:, :1])], dim=1)
                points_t = torch.matmul(curr2prev, points.T) # b, 4, n
                points_xy = points_t[:, :2].permute(0, 2, 1) # b, n, 2

                grids_xy = (points_xy - origin[:2]) / resolution
                grids_xy[:, :, 0] = 2*grids_xy[:, :, 0]/shape[-1] - 1
                grids_xy[:, :, 1] = 2*grids_xy[:, :, 1]/shape[-2] - 1
                feats = F.grid_sample(x, grids_xy.unsqueeze(1).float(), padding_mode="zeros", align_corners=False)
                x = feats.reshape(x.shape)
            warpped_bev += (x.detach(),) 

        return warpped_bev

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      img_metas=None,
                      points=None,
                      target=None):
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
        prev_occupancy = []
        occupancy = []
        prev_pts = []
        pts = []
        prev_lidar2global = []
        lidar2global = []
        for pt, img_meta in zip(points, img_metas):
            prev_pts.append(pt[-2])
            pts.append(pt[-1])
            prev_occupancy.append(img_meta['occupancy'][-2])
            occupancy.append(img_meta['occupancy'][-1])
            prev_lidar2global.append(img_meta['lidar2global'][-2])
            lidar2global.append(img_meta['lidar2global'][-1])
        
        prev_occupancy = torch.from_numpy(np.stack(prev_occupancy)).to(target.device)
        occupancy = torch.from_numpy(np.stack(occupancy)).to(target.device)
        prev_lidar2global = torch.from_numpy(np.stack(prev_lidar2global)).to(target.device)
        lidar2global = torch.from_numpy(np.stack(lidar2global)).to(target.device)

        prev_bev, _, _ = self.step(prev_pts, prev_occupancy, return_bev=True)
        if self.warpping:
            prev_bev = self.feature_warpping(prev_bev, prev_lidar2global, lidar2global)
        else:
            prev_bev = tuple([x.detach() for x in prev_bev])

        x, ss_out_dict, sc_out_dict = self.step(pts, occupancy, prev_bev=prev_bev)
        x, xg = x
        x = self.cspn(xg.squeeze(0), x.squeeze(0)).unsqueeze(0)
        loss_seg_dict = self.pts_backbone.losses(ss_out_dict, img_metas)
        loss_com_dict = self.pts_middle_encoder.losses(sc_out_dict, img_metas)
        loss_ssc = self.pts_bbox_head.losses(x, target)

        losses = {'loss_semantic_scene': loss_ssc, 'loss_semantic_seg': sum(loss_seg_dict.values()), 'loss_scene_completion': sum(loss_com_dict.values())}

        return losses

    def forward_test(self,
                     img_metas=None,
                     points=None,
                     target=None,
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
        lidar2global = torch.from_numpy(img_metas[0]['lidar2global']).to(target.device).unsqueeze(0)
        occupancy = torch.from_numpy(img_metas[0]['occupancy']).to(target.device).unsqueeze(0)
        if self.prev_info['prev_bev'] is not None and self.prev_info['sequence_id'] == img_metas[0]['sequence_id']:
            prev_bev = self.feature_warpping(self.prev_info['prev_bev'], self.prev_info['prev_lidar2global'], lidar2global)
        else:
            prev_bev = None
        outs, _, _ = self.step(points, occupancy, prev_bev=prev_bev)
        x, xg, cur_bev = outs
        x = self.cspn(xg.squeeze(0), x.squeeze(0)).unsqueeze(0)
        self.prev_info['prev_bev'] = cur_bev
        self.prev_info['sequence_id'] = img_metas[0]['sequence_id']
        self.prev_info['prev_lidar2global'] = lidar2global

        result = dict()
        result['output_voxels'] = x
        result['target_voxels'] = target

        return result
