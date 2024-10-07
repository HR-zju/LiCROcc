import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D
from mmdet.models import HEADS
from projects.mmdet3d_plugin.ssc_rs.utils.lovasz_losses import lovasz_softmax
from projects.mmdet3d_plugin.ssc_rs.utils.ssc_loss import geo_scal_loss, sem_scal_loss
from PIL import Image
from .mytransformer import CustonDATransformer
from mmcv.cnn import build_norm_layer


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
        

class BEVFusion(nn.Module):
    def __init__(self, channel, light=False, use_cam=True, use_DA=False, use_add=False):
        super().__init__()

        self.use_cam = use_cam
        self.use_add = use_add
        # if not self.use_add:
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
        self.use_DA = use_DA
        if not self.use_DA and self.use_cam and not self.use_add:
            self.attention_img = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channel, channel, kernel_size=1),
                nn.Sigmoid()
            )

        # if self.use_add and self.use_cam:
        #     # breakpoint()
        #     self.img_conv_layer = nn.Conv2d(channel*2, channel, kernel_size=1, padding=0)     
            # self.pc_conv_layer = nn.Conv2d(channel, int(channel/2.), kernel_size=1, padding=0)      
 
        
        self.light = light
        if not light:
            self.adapter_sem = nn.Conv2d(channel//2, channel, 1)
            self.adapter_com = nn.Conv2d(channel//2, channel, 1)
            if self.use_cam:
                if not self.use_add:
                    self.adapter_img = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                        nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
                        build_norm_layer(dict(type='BN'), channel)[1],
                        nn.ReLU(inplace=True),
                        nn.Conv2d(channel, channel, kernel_size=1, padding=0)
                    ) #nn.Conv2d(channel, channel, 1)
                else:
                    self.adapter_img = nn.Sequential(
                        # nn.ConvTranspose2d(channel, channel, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                        # build_norm_layer(dict(type='BN'), channel)[1],
                        )
        if self.use_DA: # 使用DA
            self.DA = CustonDATransformer(embed_dims=channel)


    def forward(self, bev_features, sem_features, com_features, img_features):
        if not self.light:
            sem_features = self.adapter_sem(sem_features)
            com_features = self.adapter_com(com_features)
            # draw_feat([img_features], 'img_features_before')

            if self.use_cam:
                img_features = self.adapter_img(img_features)
                # draw_feat([img_features], 'img_features_adp')

        # breakpoint()
        # if not self.use_add:
        attn_bev = self.attention_bev(bev_features)
        attn_sem = self.attention_sem(sem_features)
        attn_com = self.attention_com(com_features)
        if self.use_DA:
            fusion_features = torch.mul(bev_features, attn_bev) \
                + torch.mul(sem_features, attn_sem) \
                + torch.mul(com_features, attn_com) 
            fusion_features = self.DA(img_features, fusion_features)
        elif self.use_add:
            if self.use_cam:
                # attn_img = self.attention_img(img_features)
                # fusion_features = torch.add(bev_features, sem_features, com_features, img_features)
                # fusion_features = (bev_features + sem_features + com_features + img_features)

                # fusion_features = torch.mul(bev_features, attn_bev) \
                #     + torch.mul(sem_features, attn_sem) \
                #     + torch.mul(com_features, attn_com) \
                #     + img_features
                # draw_feat([torch.mul(bev_features, attn_bev) \
                #     + torch.mul(sem_features, attn_sem) \
                #     + torch.mul(com_features, attn_com)], 'bevfeat')

                fusion_features = torch.mul(bev_features, attn_bev) \
                    + torch.mul(sem_features, attn_sem) \
                    + torch.mul(com_features, attn_com) \
                    + img_features
                # fusion_features = self.pc_conv_layer(fusion_features)
                # fusion_features = torch.cat((img_features, fusion_features), dim=1)  
                # # breakpoint()
                # fusion_features = self.img_conv_layer(fusion_features)



                # breakpoint()

                # combined_tensor = torch.cat((fusion_features, img_features), dim=1)  
                # fusion_features = self.conv_layer(combined_tensor)  

                    # + torch.mul(img_features, attn_img)
            else:
                fusion_features = torch.mul(bev_features, attn_bev) \
                    + torch.mul(sem_features, attn_sem) \
                    + torch.mul(com_features, attn_com) 
        else:
            if self.use_cam:
                attn_img = self.attention_img(img_features)
                # draw_feat([attn_img], 'img_feature_atten')
                # draw_feat([torch.mul(bev_features, attn_bev) \
                #     + torch.mul(sem_features, attn_sem) \
                #     + torch.mul(com_features, attn_com)], 'bevfeat')
                fusion_features = torch.mul(bev_features, attn_bev) \
                    + torch.mul(sem_features, attn_sem) \
                    + torch.mul(com_features, attn_com) \
                    + torch.mul(img_features, attn_img)
                # fusion_features = pc_features
            else:
                fusion_features = torch.mul(bev_features, attn_bev) \
                    + torch.mul(sem_features, attn_sem) \
                    + torch.mul(com_features, attn_com) 
        # draw_feat([fusion_features], 'bev_fusion')
        # breakpoint()

        return fusion_features

############ my custom ###########
class myMLP(nn.Module):  
    def __init__(self, input_dim, hidden_dim, output_dim):  
        super(myMLP, self).__init__()  
        self.mlp = nn.Sequential(  
            nn.Linear(input_dim, hidden_dim),  
            nn.ReLU(),  
            nn.Linear(hidden_dim, output_dim)  
        )  
        self.attention_radar = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_dim, output_dim, kernel_size=1),
            nn.Sigmoid()
        )
  
    def forward(self, x):  
        # Reshape the input tensor to (B, C*H*W) before passing it through the MLP  
        attn_radar = self.attention_radar(x)
        B, C,H,W = x.size()
        # breakpoint()
        x = x.permute(0,2,3,1).reshape(-1,C)
        return self.mlp(x).reshape(B,H,W,C).permute(0,3,1,2), attn_radar

@HEADS.register_module()
class BEVUNet(nn.Module):
    def __init__(self, n_class, n_height, class_frequences=None, dilation=1, 
        bilinear=True, group_conv=False, input_batch_norm=True, dropout=0.5, circular_padding=False, dropblock=False, light=False, frozen=False, use_cam=[True,True,True], use_Distill_2 = False, use_DA=False, use_add=False, **kwargs):
        super().__init__()

        self.nbr_classes = int(n_class / n_height)
        self.n_height = n_height
        self.class_frequencies = class_frequences
        self.use_Distill_2 = use_Distill_2
        # TODO 这里的分辨率改到cfg里面
        if self.use_Distill_2:
            channels = [128, 256, 512]
            self.distill_mlp = nn.ModuleList([myMLP(channels[i], channels[i], channels[i]) for i in range(3)])

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
        self.bev_fusions = nn.ModuleList([BEVFusion(channels[i], use_cam=use_cam[i],light=light, use_DA=use_DA, use_add=use_add) for i in range(3)])
        # 冻住
        if frozen:
            for k,v in self.named_parameters():
                # if 'tgms' not in k:
                v.requires_grad = False

    def forward(self, x, sem_fea_list, com_fea_list, img_feats_list):

        x1 = self.inc(x)    
        x2 = self.down1(x1)   
        x2_f = self.bev_fusions[0](x2, sem_fea_list[0], com_fea_list[0], img_feats_list[0])
        if self.use_Distill_2:
            x2_f_residue, x2_f_atten = self.distill_mlp[0](x2_f)
            x2_f = torch.add(x2_f, torch.mul(x2_f_residue, x2_f_atten))

        x3 = self.down2(x2_f)    # [B, 256, 64, 64]
        x3_f = self.bev_fusions[1](x3, sem_fea_list[1], com_fea_list[1], img_feats_list[1]) 
        if self.use_Distill_2:
            x3_f_residue, x3_f_atten = self.distill_mlp[1](x3_f)
            x3_f = torch.add(x3_f, torch.mul(x3_f_residue, x3_f_atten))
        x4 = self.down3(x3_f)   
        x4_f = self.bev_fusions[2](x4, sem_fea_list[2], com_fea_list[2], img_feats_list[2]) 
        if self.use_Distill_2:
            x4_f_residue, x4_f_atten = self.distill_mlp[2](x4_f)
            x4_f = torch.add(x4_f, torch.mul(x4_f_residue, x4_f_atten))
        x5 = self.down4(x4_f)   
        x = self.up1(x5, x4_f)  
        x = self.up2(x, x3_f)  
        x = self.up3(x, x2_f)  
        x = self.up4(x, x1)  
        x = self.outc(self.dropout(x))

        new_shape = [x.shape[0], self.nbr_classes, self.n_height, *x.shape[-2:]]   
        x = x.view(new_shape)
        x = x.permute(0,1,4,3,2)  

        if self.use_Distill_2:
            return x, [x2_f, x3_f, x4_f, x5], [x2_f_residue, x3_f_residue, x4_f_residue]
        else:
            return x, [x2_f, x3_f, x4_f, x5], None

    # 
    def losses(self, scores, labels):
        class_weights = torch.from_numpy(1 / np.log(np.array(self.class_frequencies) + 0.001)).float().to(scores.device)
        loss = F.cross_entropy(scores, labels.long(), weight=class_weights, ignore_index=255)
       
        loss_sem_scal = sem_scal_loss(scores, labels.long())
        loss += loss_sem_scal
        
        loss_geo_scal = geo_scal_loss(scores, labels.long())

        loss += loss_geo_scal

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
