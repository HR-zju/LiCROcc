import torch
import torch.nn as nn


class Affinity_Propagate(nn.Module):

    def __init__(self,
                 prop_time,
                 norm_type='8sum'):
        """
        Inputs:
            prop_time: how many steps for CSPN to perform
            way to normalize affinity
                '8sum': normalize using 8 surrounding neighborhood
                '8sum_abs': normalization enforcing affinity to be positive
                            This will lead the center affinity to be 0
        """
        super(Affinity_Propagate, self).__init__()
        self.prop_time = prop_time
        self.norm_type = norm_type
        assert norm_type in ['8sum', '8sum_abs']

        self.in_feature = 1
        self.out_feature = 1

    def gen_operator(self):
        self.sum_conv = nn.Conv3d(in_channels=8,
                                  out_channels=1,
                                  kernel_size=(1, 1, 1),
                                  stride=1,
                                  padding=0,
                                  bias=False)
        weight = torch.ones(1, 8, 1, 1, 1).cuda()
        self.sum_conv.weight = nn.Parameter(weight)
        for param in self.sum_conv.parameters():
            param.requires_grad = False

    def propagate_once(self, gate_wb, gate_sum, blur):
        result = self.pad_blur(blur)
        neigbor_weighted_sum = self.sum_conv(gate_wb * result)
        neigbor_weighted_sum = neigbor_weighted_sum.squeeze(1)
        neigbor_weighted_sum = neigbor_weighted_sum[:, :, 1:-1, 1:-1]
        result = neigbor_weighted_sum

        if '8sum' in self.norm_type:
                result = (1.0 - gate_sum) * blur + result
        else:
            raise ValueError('unknown norm %s' % self.norm_type)

        return result

    def forward(self, guidance, blur):
        self.gen_operator()

        guidance_xyz = torch.split(guidance, 8, dim=0) # 24, h, w, z

        gate_wb_x, gate_sum_x = self.affinity_normalization(guidance_xyz[0].permute(3, 0, 1, 2))
        gate_wb_y, gate_sum_y = self.affinity_normalization(guidance_xyz[1].permute(2, 0, 1, 3))
        gate_wb_z, gate_sum_z = self.affinity_normalization(guidance_xyz[2].permute(1, 0, 2, 3))

        result = blur  # 20, h, w, z
        for _ in range(self.prop_time):
            # one propagation
            result = self.propagate_once(gate_wb_x, gate_sum_x, result.permute(3, 0, 1, 2)).permute(1, 2, 3, 0)
            result = self.propagate_once(gate_wb_y, gate_sum_y, result.permute(2, 0, 1, 3)).permute(1, 2, 0, 3)
            result = self.propagate_once(gate_wb_z, gate_sum_z, result.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)

        return result

    def affinity_normalization(self, guidance):

        # normalize features
        if 'abs' in self.norm_type:
            guidance = torch.abs(guidance)

        gate1_wb_cmb = guidance.narrow(1, 0                   , self.out_feature)
        gate2_wb_cmb = guidance.narrow(1, 1 * self.out_feature, self.out_feature)
        gate3_wb_cmb = guidance.narrow(1, 2 * self.out_feature, self.out_feature)
        gate4_wb_cmb = guidance.narrow(1, 3 * self.out_feature, self.out_feature)
        gate5_wb_cmb = guidance.narrow(1, 4 * self.out_feature, self.out_feature)
        gate6_wb_cmb = guidance.narrow(1, 5 * self.out_feature, self.out_feature)
        gate7_wb_cmb = guidance.narrow(1, 6 * self.out_feature, self.out_feature)
        gate8_wb_cmb = guidance.narrow(1, 7 * self.out_feature, self.out_feature)

        # gate1:left_top, gate2:center_top, gate3:right_top
        # gate4:left_center,              , gate5: right_center
        # gate6:left_bottom, gate7: center_bottom, gate8: right_bottm

        # top pad
        left_top_pad = nn.ZeroPad2d((0,2,0,2))
        gate1_wb_cmb = left_top_pad(gate1_wb_cmb).unsqueeze(1)

        center_top_pad = nn.ZeroPad2d((1,1,0,2))
        gate2_wb_cmb = center_top_pad(gate2_wb_cmb).unsqueeze(1)

        right_top_pad = nn.ZeroPad2d((2,0,0,2))
        gate3_wb_cmb = right_top_pad(gate3_wb_cmb).unsqueeze(1)

        # center pad
        left_center_pad = nn.ZeroPad2d((0,2,1,1))
        gate4_wb_cmb = left_center_pad(gate4_wb_cmb).unsqueeze(1)

        right_center_pad = nn.ZeroPad2d((2,0,1,1))
        gate5_wb_cmb = right_center_pad(gate5_wb_cmb).unsqueeze(1)

        # bottom pad
        left_bottom_pad = nn.ZeroPad2d((0,2,2,0))
        gate6_wb_cmb = left_bottom_pad(gate6_wb_cmb).unsqueeze(1)

        center_bottom_pad = nn.ZeroPad2d((1,1,2,0))
        gate7_wb_cmb = center_bottom_pad(gate7_wb_cmb).unsqueeze(1)

        right_bottm_pad = nn.ZeroPad2d((2,0,2,0))
        gate8_wb_cmb = right_bottm_pad(gate8_wb_cmb).unsqueeze(1)

        gate_wb = torch.cat((gate1_wb_cmb,gate2_wb_cmb,gate3_wb_cmb,gate4_wb_cmb,
                             gate5_wb_cmb,gate6_wb_cmb,gate7_wb_cmb,gate8_wb_cmb), 1)

        # normalize affinity using their abs sum
        gate_wb_abs = torch.abs(gate_wb)
        abs_weight = self.sum_conv(gate_wb_abs)

        gate_wb = torch.div(gate_wb, abs_weight)
        gate_sum = self.sum_conv(gate_wb)

        gate_sum = gate_sum.squeeze(1)
        gate_sum = gate_sum[:, :, 1:-1, 1:-1]

        return gate_wb, gate_sum


    def pad_blur(self, blur):
        # top pad
        left_top_pad = nn.ZeroPad2d((0,2,0,2))
        blur_1 = left_top_pad(blur).unsqueeze(1)
        center_top_pad = nn.ZeroPad2d((1,1,0,2))
        blur_2 = center_top_pad(blur).unsqueeze(1)
        right_top_pad = nn.ZeroPad2d((2,0,0,2))
        blur_3 = right_top_pad(blur).unsqueeze(1)

        # center pad
        left_center_pad = nn.ZeroPad2d((0,2,1,1))
        blur_4 = left_center_pad(blur).unsqueeze(1)
        right_center_pad = nn.ZeroPad2d((2,0,1,1))
        blur_5 = right_center_pad(blur).unsqueeze(1)

        # bottom pad
        left_bottom_pad = nn.ZeroPad2d((0,2,2,0))
        blur_6 = left_bottom_pad(blur).unsqueeze(1)
        center_bottom_pad = nn.ZeroPad2d((1,1,2,0))
        blur_7 = center_bottom_pad(blur).unsqueeze(1)
        right_bottm_pad = nn.ZeroPad2d((2,0,2,0))
        blur_8 = right_bottm_pad(blur).unsqueeze(1)

        result_depth = torch.cat((blur_1, blur_2, blur_3, blur_4,
                                  blur_5, blur_6, blur_7, blur_8), 1)
        return result_depth