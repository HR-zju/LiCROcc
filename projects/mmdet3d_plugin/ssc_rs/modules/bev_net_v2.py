import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D
from mmdet.models import HEADS
from projects.mmdet3d_plugin.ssc_rs.utils.lovasz_losses import lovasz_softmax
from projects.mmdet3d_plugin.ssc_rs.utils.ssc_loss import geo_scal_loss, sem_scal_loss
from projects.mmdet3d_plugin.ssc_rs.modules.ops.modules import MSDeformAttn


class BEVFusion(nn.Module):
    def __init__(self, channel, light=False):
        super().__init__()

        self.attention_bev = nn.Sequential(
             nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.Sigmoid()
        )
        self.attention_sem = nn.Sequential(
             nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.Sigmoid()
        )
        self.attention_com = nn.Sequential(
             nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.Sigmoid()
        )

        self.light = light
        if not light:
            self.adapter_sem = nn.Conv2d(channel//2, channel, 1)
            self.adapter_com = nn.Conv2d(channel//2, channel, 1)

    def forward(self, bev_features, sem_features, com_features):
        if not self.light:
            sem_features = self.adapter_sem(sem_features)
            com_features = self.adapter_com(com_features
            )
        attn_bev = self.attention_bev(bev_features)
        attn_sem = self.attention_sem(sem_features)
        attn_com = self.attention_com(com_features)

        fusion_features = torch.mul(bev_features, attn_bev) \
            + torch.mul(sem_features, attn_sem) \
            + torch.mul(com_features, attn_com)

        return fusion_features


class TemporalGuidedModule(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv_in_1 = nn.Conv2d(channel, channel, kernel_size=1)
        self.conv_in_2 = nn.Conv2d(channel, channel, kernel_size=1)
        self.cross_attn = MSDeformAttn(d_model=channel, n_levels=1, n_heads=8, n_points=8)
        self.conv_out = nn.Conv2d(channel, channel, kernel_size=1)

        self.temporal_guided = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1)
        )

    def forward(self, xt, xt_1):
        if xt_1 is None:
            xt_1 = xt

        xt = self.conv_in_1(xt)
        xt_1 = self.conv_in_2(xt_1)
        bs, c, h, w = xt.shape
        spatial_shape = torch.as_tensor((h, w), dtype=torch.long, device=xt.device)[None] # 1, 2
        level_start_index = torch.cat((spatial_shape.new_zeros((1, )), spatial_shape.prod(1).cumsum(0)[:-1]))

        yv, xv = np.meshgrid(range(h), range(w), indexing='ij')
        grids = torch.from_numpy(np.concatenate([(xv.reshape(1,-1)+0.5), (yv.reshape(1,-1)+0.5)], axis=0)).float().T
        grids_xy = grids.expand(bs, -1, -1).contiguous().to(xt.device)
        reference_points = grids_xy / torch.as_tensor((w, h), dtype=torch.float32, device=xt.device)[None, None]
        reference_points = reference_points[:, :, None]

        xt_flatten = xt.permute(0, 2, 3, 1).flatten(1, 2) # b, n, c
        xt_1_flatten = xt_1.permute(0, 2, 3, 1).flatten(1, 2) # b, n, c
        xt_star = self.cross_attn(xt_flatten, reference_points, xt_1_flatten, spatial_shape, level_start_index)
        xt_star = xt_star.permute(0, 2, 1).reshape(bs, c, h, w)

        xt = xt + self.temporal_guided(xt * xt_star)
        xt = F.relu(self.conv_out(xt))

        return xt


@HEADS.register_module()
class BEVUNetV2(nn.Module):
    def __init__(self, n_class, n_height, class_frequences=None, dilation=1, 
        bilinear=True, group_conv=False, input_batch_norm=True, dropout=0.5, circular_padding=False, dropblock=False, light=False, target_scales=['1_16'], **kwargs):
        super().__init__()

        self.nbr_classes = int(n_class / n_height)
        self.n_height = n_height
        self.class_frequencies = class_frequences

        self.inc = inconv(n_height*2, 64, dilation, input_batch_norm, circular_padding)
        if not light:
            self.down1 = down(64, 128, dilation, group_conv, circular_padding)
            self.down2 = down(128, 256, dilation, group_conv, circular_padding)
            self.down3 = down(256, 512, dilation, group_conv, circular_padding)
            self.down4 = down(512, 512, dilation, group_conv, circular_padding)
            self.up1 = up(1024, 512, circular_padding, bilinear = bilinear, group_conv = group_conv, use_dropblock=dropblock, drop_p=dropout)
            self.up2 = up(768, 256, circular_padding, bilinear = bilinear, group_conv = group_conv, use_dropblock=dropblock, drop_p=dropout)
            self.up3 = up(384, 128, circular_padding, bilinear = bilinear, group_conv = group_conv, use_dropblock=dropblock, drop_p=dropout)
            self.up4 = up(192, 128, circular_padding, bilinear = bilinear, group_conv = group_conv, use_dropblock=dropblock, drop_p=dropout)
            self.dropout = nn.Dropout(p=0. if dropblock else dropout)
            self.outc = outconv(128, n_class)

            channels = [128, 256, 512]
        else:
            self.down1 = down(64, 64, dilation, group_conv, circular_padding)
            self.down2 = down(64, 128, dilation, group_conv, circular_padding)
            self.down3 = down(128, 256, dilation, group_conv, circular_padding)
            self.down4 = down(256, 256, dilation, group_conv, circular_padding)
            self.up1 = up(512, 256, circular_padding, bilinear = bilinear, group_conv = group_conv, use_dropblock=dropblock, drop_p=dropout)
            self.up2 = up(384, 128, circular_padding, bilinear = bilinear, group_conv = group_conv, use_dropblock=dropblock, drop_p=dropout)
            self.up3 = up(192, 64, circular_padding, bilinear = bilinear, group_conv = group_conv, use_dropblock=dropblock, drop_p=dropout)
            self.up4 = up(128, 64, circular_padding, bilinear = bilinear, group_conv = group_conv, use_dropblock=dropblock, drop_p=dropout)
            self.dropout = nn.Dropout(p=0. if dropblock else dropout)
            self.outc = outconv(64, n_class)

            channels = [64, 128, 256]
        self.bev_fusions = nn.ModuleList([BEVFusion(channels[i], light=light) for i in range(3)])

        self.target_scales = target_scales
        module_dict = {}
        ch_dict = {'1_2': 64 if light else 128, '1_4': 128 if light else 256, '1_8': 256 if light else 512, '1_16': 256 if light else 512}
        for s in self.target_scales:
            module_dict.update({s: TemporalGuidedModule(channel=ch_dict[s])})
        self.tgms = nn.ModuleDict(module_dict)

    def forward(self, x, sem_fea_list, com_fea_list, return_bev=False, prev_bev=None):
        # sptail bev fusion
        cur_bev = tuple()
        x1 = self.inc(x)    # [B, 64, 256, 256]
        x2 = self.down1(x1)    # [B, 128, 128, 128]
        x2_f = self.bev_fusions[0](x2, sem_fea_list[0], com_fea_list[0]) # 128, 64, 64 -> 128
        cur_bev += (x2_f, )

        x3 = self.down2(x2_f)    # [B, 256, 64, 64]
        x3_f = self.bev_fusions[1](x3, sem_fea_list[1], com_fea_list[1]) # 256, 128, 128 -> 256
        cur_bev += (x3_f, )

        x4 = self.down3(x3_f)    # [B, 512, 32, 32]
        x4_f = self.bev_fusions[2](x4, sem_fea_list[2], com_fea_list[2]) # 512, 256, 256 -> 512
        cur_bev += (x4_f, )

        x5 = self.down4(x4_f)    # [B, 512, 16, 16]
        cur_bev += (x5, )

        if return_bev:
            return cur_bev

        # tempral bev fusion
        prev_bev_x2, prev_bev_x3, prev_bev_x4, prev_bev_x5 = prev_bev if isinstance(prev_bev, tuple) else [None for _ in range(4)]
        x2_t = self.tgms['1_2'](x2_f, prev_bev_x2) if '1_2' in self.target_scales else x2_f
        x3_t = self.tgms['1_4'](x3_f, prev_bev_x3) if '1_4' in self.target_scales else x3_f
        x4_t = self.tgms['1_8'](x4_f, prev_bev_x4) if '1_8' in self.target_scales else x4_f
        x5_t = self.tgms['1_16'](x5, prev_bev_x5) if '1_16' in self.target_scales else x5

        x = self.up1(x5_t, x4_t)  # 512, 512
        x = self.up2(x, x3_t)  # 512, 256
        x = self.up3(x, x2_t)  # 256, 128
        x = self.up4(x, x1)  # 128, 64
        x = self.outc(self.dropout(x))

        new_shape = [x.shape[0], self.nbr_classes, self.n_height, *x.shape[-2:]]    # [B, 20, 32, 256, 256]
        x = x.view(new_shape)
        x = x.permute(0,1,4,3,2)   # [B,20,256,256,32]
        
        if not self.training:
            return x, cur_bev
        return x

    def losses(self, scores, labels):
        class_weights = torch.from_numpy(1 / np.log(np.array(self.class_frequencies) + 0.001)).float().to(scores.device)
        loss = F.cross_entropy(scores, labels.long(), weight=class_weights, ignore_index=255)
        loss += lovasz_softmax(torch.nn.functional.softmax(scores, dim=1), labels.long(), ignore=255)
        # loss_sem_scal = sem_scal_loss(scores, labels.long())
        # loss += loss_sem_scal

        # loss_geo_scal = geo_scal_loss(scores, labels.long())
        # loss += loss_geo_scal
        loss = loss*3

        return loss


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch,group_conv,dilation=1):
        super(double_conv, self).__init__()
        if group_conv:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1,groups = min(out_ch,in_ch)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1,groups = out_ch),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x

class double_conv_circular(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch,group_conv,dilation=1):
        super(double_conv_circular, self).__init__()
        if group_conv:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=(1,0),groups = min(out_ch,in_ch)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=(1,0),groups = out_ch),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=(1,0)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=(1,0)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )

    def forward(self, x):
        #add circular padding
        x = F.pad(x,(1,1,0,0),mode = 'circular')
        x = self.conv1(x)
        x = F.pad(x,(1,1,0,0),mode = 'circular')
        x = self.conv2(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, input_batch_norm, circular_padding):
        super(inconv, self).__init__()
        if input_batch_norm:
            if circular_padding:
                self.conv = nn.Sequential(
                    nn.BatchNorm2d(in_ch),
                    double_conv_circular(in_ch, out_ch,group_conv = False,dilation = dilation)
                )
            else:
                self.conv = nn.Sequential(
                    nn.BatchNorm2d(in_ch),
                    double_conv(in_ch, out_ch,group_conv = False,dilation = dilation)
                )
        else:
            if circular_padding:
                self.conv = double_conv_circular(in_ch, out_ch,group_conv = False,dilation = dilation)
            else:
                self.conv = double_conv(in_ch, out_ch,group_conv = False,dilation = dilation)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, group_conv, circular_padding):
        super(down, self).__init__()
        if circular_padding:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv_circular(in_ch, out_ch,group_conv = group_conv,dilation = dilation)
            )
        else:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_ch, out_ch, group_conv=group_conv, dilation=dilation)
            )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, circular_padding, bilinear=True, group_conv=False, use_dropblock=False, drop_p=0.5):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif group_conv:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2, groups = in_ch//2)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        if circular_padding:
            self.conv = double_conv_circular(in_ch, out_ch,group_conv = group_conv)
        else:
            self.conv = double_conv(in_ch, out_ch, group_conv = group_conv)

        self.use_dropblock = use_dropblock
        if self.use_dropblock:
            self.dropblock = DropBlock2D(block_size=7, drop_prob=drop_p)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        if self.use_dropblock:
            x = self.dropblock(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
