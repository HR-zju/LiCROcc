import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from mmdet3d.models.builder import MIDDLE_ENCODERS

from projects.mmdet3d_plugin.ssc_rs.utils.lovasz_losses import lovasz_softmax
from PIL import Image

class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, padding, stride, dilation=1):
        super().__init__()
        self.reduction = nn.Conv3d(in_dim, out_dim, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)
        self.layer = nn.Conv3d(out_dim, out_dim, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)

    def forward(self, x):
        add = self.reduction(x)
        out = self.layer(F.relu(add))
        out_res = F.relu(add + out)
        return out_res


def make_layers(in_dim, out_dim, kernel_size=3, padding=1, stride=1, dilation=1,downsample=False, blocks=2):
    layers = []
    if downsample:
        layers.append(nn.MaxPool3d(2))
    layers.append(ResBlock(in_dim, out_dim, kernel_size, padding, stride, dilation))
    for _ in range(1, blocks):
        layers.append(ResBlock(out_dim, out_dim, kernel_size, padding, stride, dilation))
    return nn.Sequential(*layers)


@MIDDLE_ENCODERS.register_module()
class CompletionBranch(nn.Module):
    def __init__(self, init_size=32, nbr_class=20, phase='trainval', frozen = False):
        super().__init__()
        self.nclass = nbr_class
        self.in_layer =  nn.Conv3d(1, 16, kernel_size=7, padding=3, stride=2, dilation=1)  # 1/2, 16
        self.block_1 = make_layers(16, 16, kernel_size=3, padding=1, stride=1, dilation=1, blocks=1) # 1/2, 16
        self.block_2 = make_layers(16, 32, kernel_size=3, padding=1, stride=1, dilation=1, downsample=True, blocks=1) # 1/4, 32
        self.block_3 = make_layers(32, 64, kernel_size=3, padding=2, stride=1, dilation=2, downsample=True, blocks=1)  # 1/8, 64

        self.reduction_1 = nn.Sequential(
            nn.Conv2d(16*init_size//2, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU()
        )
        self.reduction_2 = nn.Sequential(
            nn.Conv2d(16*init_size//2, 128, kernel_size=1),
            nn.ReLU(),
        )

        self.init_size = init_size
        if init_size == 40:
            self.reduction_3 = nn.Sequential(
                nn.Conv2d(16*init_size//2, 256, kernel_size=1),
                nn.ReLU()
            )

        self.phase = phase
        if phase == 'trainval':
            self.out2 = nn.Sequential(
                nn.Conv3d(16, 16, kernel_size=1),
                nn.ReLU(),
                nn.Conv3d(16, 2, kernel_size=1))
            self.out4 = nn.Sequential(
                nn.Conv3d(32, 32, kernel_size=1),
                nn.ReLU(),
                nn.Conv3d(32, 2, kernel_size=1))
            self.out8 = nn.Sequential(
                nn.Conv3d(64, 32, kernel_size=1),
                nn.ReLU(),
                nn.Conv3d(32, 2, kernel_size=1))
        
        if frozen:
            for k,v in self.named_parameters():
                # if 'tgms' not in k:
                v.requires_grad = False

    def forward(self, inputs):
        out = F.relu(self.in_layer(inputs))
        res1 = self.block_1(out)  # B, 16, 16, 128, 128
        res2 = self.block_2(res1)  # B, 32, 8, 64, 64
        res3 = self.block_3(res2)  # B, 64, 4, 32, 32

        bev_1 = self.reduction_1(res1.flatten(1, 2)) # B, 64, 128, 128
        bev_2 = self.reduction_2(res2.flatten(1, 2)) # B, 128, 64, 64
        if self.init_size == 40:
            bev_3 = self.reduction_3(res3.flatten(1, 2)) 
        else:
            bev_3 = res3.flatten(1, 2) # B, 256, 32, 32

        return dict(
            mss_bev_dense = [bev_1, bev_2, bev_3],
            mss_logits = [self.out2(res1), self.out4(res2), self.out8(res3)] if self.phase == 'trainval' else None
        )

    def losses(self, out_dict, metas):
        mss_logits = out_dict['mss_logits']
        device = mss_logits[0].device

        loss_dict = {}
        for i in range(len(mss_logits)):
            target = []
            for x in metas:
                target.append(x[f'target_1_{2**(i+1)}'])
            target = torch.from_numpy(np.stack(target)).to(device)
            labels_copy = target.long().clone()
            valid = (0 < labels_copy) & (labels_copy < self.nclass)
            labels_copy[valid] = 1
            
            scores = mss_logits[i].permute(0, 1, 4, 3, 2) # BCDHW
            # scores = mss_logits[i].permute(0, 1, 3, 4, 2) # BCDHW

            # 画图
            # mean_along_128 = labels_copy.float().mean(dim=-1)[0]
            # # mean_along_128 = mean_along_128.mean(dim=1)[0]
            # image_tensor = mean_along_128
            # # image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())  
            # image_array = image_tensor.detach().cpu().numpy()  
            # image = Image.fromarray(np.uint8(image_array * 255))  
            # image.save("score_0.png")  

            # mean_along_128 = scores.float().mean(dim=1)
            # mean_along_128 = mean_along_128.float().mean(dim=-1)[0]

            # # mean_along_128 = mean_along_128.mean(dim=1)[0]
            # image_tensor = mean_along_128
            # image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())  
            # image_array = image_tensor.detach().cpu().numpy()  
            # image = Image.fromarray(np.uint8(image_array * 255))  
            # image.save("com_score_0.png") 
            # breakpoint()

            scale_loss = lovasz_softmax(F.softmax(scores, dim=1), labels_copy, ignore=255)
            focal_loss = F.cross_entropy(scores, labels_copy, ignore_index=255)
            loss_dict["ds_" + str(i) + "lovasz_loss"] = scale_loss
            loss_dict["ds_" + str(i) + "ce_loss"] = focal_loss

        return loss_dict

