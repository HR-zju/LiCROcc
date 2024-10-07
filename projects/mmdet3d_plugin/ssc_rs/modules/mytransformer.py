import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from mmcv.cnn import constant_init, xavier_init
import copy
import spconv.pytorch as spconv

'''
Transformer位置编码
'''
class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Linear(input_channel, num_pos_feats, bias=False),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Linear(num_pos_feats, num_pos_feats, bias=False))
        
        xavier_init(self.position_embedding_head, distribution='uniform', bias=0.)
        
    def forward(self, xyz):
        b, q, c = xyz.shape
        xyz = xyz.view(b*q, c)
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding.view(b, q, -1)

class CustonDATransformer(nn.Module):
    
    def __init__(self, 
                 num_layer = 6,
                 embed_dims=512,
                 num_heads=8,
                 num_levels=1,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        """_summary_

        Args:
            num_layer (int, optional): num of CustonDATransformerLayer. Defaults to 6.
            embed_dims (int, optional): _description_. Defaults to 256.
            num_heads (int, optional): _description_. Defaults to 8.
            num_levels (int, optional): _description_. Defaults to 4.
            num_points (int, optional): _description_. Defaults to 4.
            im2col_step (int, optional): _description_. Defaults to 64.
            dropout (float, optional): _description_. Defaults to 0.1.
            batch_first (bool, optional): _description_. Defaults to False.
            norm_cfg (_type_, optional): _description_. Defaults to None.
            init_cfg (_type_, optional): _description_. Defaults to None.
        """
        super(CustonDATransformer, self).__init__()
        self.num_levels = num_levels
        encoder_layer = CustonDATransformerLayer(embed_dims=embed_dims,
                                                    num_heads=num_heads,
                                                    num_levels=num_levels,
                                                    num_points=num_points,
                                                    im2col_step=im2col_step,
                                                    dropout=dropout,
                                                    batch_first=batch_first,
                                                    norm_cfg=norm_cfg,
                                                    init_cfg=init_cfg)
        self.layers = []
        for i in range(num_layer):
            self.layers.append(copy.deepcopy(encoder_layer))
        self.layers = nn.ModuleList(self.layers)
        self.posembed = PositionEmbeddingLearned(2, embed_dims) # 手动给了512的位置编码

    def forward(self, bev_feat, lidar_feat):
        # print(bev_feat.shape, lidar_feat.features.shape, lidar_feat.indices.shape, lidar_feat.spatial_shape)
        # breakpoint()
        B, C, H, W = bev_feat.shape
        # breakpoint()
        lidar_feat = lidar_feat.permute(0,2,3,1).reshape(B*H*W, C)

        b = torch.arange(B)  
        h = torch.arange(H)  
        w = torch.arange(W)  
        
        # 使用 torch.meshgrid 创建三个张量，分别对应B、H和W三个维度上的坐标  
        B_coord, H_coord, W_coord = torch.meshgrid(b, h, w)  
        
        # 将这些张量转换为坐标张量  
        coordinates = torch.stack((B_coord, H_coord, W_coord), dim=-1).reshape(B*H*W, 3)
        nonzero_indices = (lidar_feat != 0).all(dim=1).nonzero()  

        
        lidar_feat = spconv.SparseConvTensor(lidar_feat[nonzero_indices].squeeze(1), coordinates[nonzero_indices].squeeze(1).int(), [H,W], B)
        # breakpoint()

        # reference_points
        ref_x = lidar_feat.indices[..., 2] / lidar_feat.spatial_shape[1]  # 归一
        ref_y = lidar_feat.indices[..., 1] / lidar_feat.spatial_shape[0]
        ref_2d = torch.stack((ref_x, ref_y), -1)  # 

        # q的batch处理成一样长的        
        mask_each = []
        for i in range(B):
            mask = (lidar_feat.indices[:, 0] == i)
            mask_each.append(mask)
        mask_each_num = [torch.sum(mask) for mask in mask_each]       
        num_q_max = max(mask_each_num)
        
        queries_rebatch = lidar_feat.features.new_zeros([B, num_q_max, C])
        reference_points_rebatch = ref_2d.new_zeros([B, num_q_max, 2]).to(bev_feat.device)
        for i in range(B):
            queries_rebatch[i, 0:mask_each_num[i], ...] = lidar_feat.features[mask_each[i]]
            reference_points_rebatch[i, 0:mask_each_num[i], ...] = ref_2d[mask_each[i]]
        q_pose = self.posembed(reference_points_rebatch)
        reference_points_rebatch = reference_points_rebatch.unsqueeze(2).repeat(1, 1, self.num_levels, 1)  # bs, num_query, num_levels, 2
        # print(queries_rebatch.shape, reference_points_rebatch.shape)
         
        # value
        feat_flatten = bev_feat.flatten(2).permute(0, 2, 1)    # [B, h*w, c]
        spatial_shapes = torch.as_tensor([(H, W)], dtype=torch.long, device=bev_feat.device)  # [L, 2]
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))  # [L, ]
        # print(feat_flatten.shape, spatial_shapes, level_start_index)
        
        # breakpoint()
        for layer in self.layers:
            queries_rebatch = layer(queries_rebatch, feat_flatten, reference_points_rebatch, spatial_shapes, level_start_index, q_pose)
        
        slots = torch.zeros_like(lidar_feat.features)
        for i in range(B):
            slots[mask_each[i]] = queries_rebatch[i, 0:mask_each_num[i], ...]
        lidar_feat = lidar_feat.replace_feature(slots)
        # breakpoint()
        return lidar_feat.dense()
        


class CustonDATransformerLayer(nn.Module):
    
    def __init__(self, 
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        """_summary_

        Args:
            embed_dims (int, optional): _description_. Defaults to 256.
            num_heads (int, optional): _description_. Defaults to 8.
            num_levels (int, optional): _description_. Defaults to 4.
            num_points (int, optional): _description_. Defaults to 4.
            im2col_step (int, optional): _description_. Defaults to 64.
            dropout (float, optional): _description_. Defaults to 0.1.
            batch_first (bool, optional): _description_. Defaults to False.
            norm_cfg (_type_, optional): _description_. Defaults to None.
            init_cfg (_type_, optional): _description_. Defaults to None.
        """
        super(CustonDATransformerLayer, self).__init__()
        self.attention = MultiScaleDeformableAttention(embed_dims=embed_dims,
                                                        num_heads=num_heads,
                                                        num_levels=num_levels,
                                                        num_points=num_points,
                                                        im2col_step=im2col_step,
                                                        dropout=dropout,
                                                        batch_first=batch_first,
                                                        norm_cfg=norm_cfg,
                                                        init_cfg=init_cfg)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims, bias=False),
            nn.ReLU(True),
            nn.Linear(embed_dims, embed_dims, bias=False),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dims)
        
        xavier_init(self.ffn, distribution='uniform', bias=0.)

    def forward(self, lidar_feat, feat_flatten, ref_2d, spatial_shapes, level_start_index, q_pose):
        queries = self.attention(query=lidar_feat, key=feat_flatten, value=feat_flatten, query_pos = q_pose,
                                 reference_points=ref_2d, spatial_shapes=spatial_shapes,level_start_index=level_start_index)
        queries = self.norm1(queries)
        queries = self.ffn(queries) + self.dropout(queries)
        queries = self.norm2(queries)
        return queries