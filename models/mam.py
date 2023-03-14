from bdb import set_trace
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import imageio
import time
import math
from torch.nn.parameter import Parameter
from einops import rearrange


class Correlation_Module(nn.Module):
    def __init__(self, in_channel):
        super(Correlation_Module, self).__init__()
        self.in_channel = in_channel
        mid_feature = in_channel // 2
        self.conva = nn.Conv1d(in_channel, mid_feature, kernel_size=1, bias=False)
        self.convb = nn.Conv1d(in_channel, mid_feature, kernel_size=1, bias=False)
        self.convc = nn.Conv1d(in_channel, mid_feature, kernel_size=1, bias=False)
        
        self.convn = nn.Conv1d(mid_feature, mid_feature, kernel_size=1, bias=False)
        self.convl = nn.Conv1d(mid_feature, mid_feature, kernel_size=1, bias=False)
        self.convd = nn.Sequential(
                nn.Conv1d(mid_feature * 2, in_channel, kernel_size=1, bias=False),
                nn.BatchNorm1d(in_channel)
                )
        
        self.line_conv_att = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)

    def forward(self, x, curves):
        curves_att = self.line_conv_att(curves)  # bs, 1, c_n, c_l

        curver_inter = torch.sum(curves * F.softmax(curves_att, dim=-1), dim=-1)  #bs, c, c_n
        curves_intra = torch.sum(curves * F.softmax(curves_att, dim=-2), dim=-2)  #bs, c, c_l

        curver_inter = self.conva(curver_inter) # bs, mid, n
        curves_intra = self.convb(curves_intra) # bs, mid, n

        x_logits = self.convc(x).transpose(1, 2).contiguous()
        x_inter = F.softmax(torch.bmm(x_logits, curver_inter), dim=-1) # bs, n, c_n
        x_intra = F.softmax(torch.bmm(x_logits, curves_intra), dim=-1) # bs, l, c_l
        

        curver_inter = self.convn(curver_inter).transpose(1, 2).contiguous()
        curves_intra = self.convl(curves_intra).transpose(1, 2).contiguous()

        x_inter = torch.bmm(x_inter, curver_inter)
        x_intra = torch.bmm(x_intra, curves_intra)

        curve_features = torch.cat((x_inter, x_intra),dim=-1).transpose(1, 2).contiguous()
        x = x + self.convd(curve_features)

        return F.leaky_relu(x, negative_slope=0.2)


class Motion_Aggregation_Module(nn.Module):
    def __init__(self, in_channels, k, num_motion):
        super(Motion_Aggregation_Module, self).__init__()
        self.Corr = Correlation_Module(in_channels)
        self.linear = nn.Linear(64, 32)
        self.num_motion = num_motion
        
        self.conv = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(in_channels),
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x_global, x_local):
        # B, C, N
        # B, C, N, L

        # N_ray, N_motion, 32
        # N_ray * N_motion, N_sample, N_dim

        x_local = rearrange(x_local, '(N_ray N_motion) N_sample N_dim -> N_ray N_motion N_sample N_dim', N_motion=(self.num_motion+1))
        x_local = self.linear(x_local)
        x_local = rearrange(x_local, 'B N L C -> B C N L')

        x_global = rearrange(x_global, 'B N C -> B C N')

        result = self.Corr(x_global, x_local)
        result = rearrange(result, 'B C N -> B N C')

        return result

