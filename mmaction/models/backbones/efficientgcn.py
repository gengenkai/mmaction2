import torch
import torch.nn as nn
from torch import nn
from mmcv.cnn import constant_init, kaiming_init, normal_init
from mmcv.runner import load_checkpoint
from mmcv.utils import _BatchNorm

from ...utils import get_root_logger
from ..builder import BACKBONES
from ..skeleton_gcn.utils import Graph
import math 

# def rescale_block(block_args, scale_args, scale_factor):
#     channel_scaler = math.pow(scale_args[0], scale_factor)
#     depth_scaler = math.pow(scale_args[1], scale_factor)
#     new_block_args = []
#     for [channel, stride, depth] in block_args:
#         channel = max(int(round(channel * channel_scaler / 16)) * 16, 16)
#         depth = int(round(depth * depth_scaler))
#         new_block_args.append([channel, stride, depth])
#     return new_block_args

# block_args = [[48,1,0.5],[24,1,0.5],[64,2,1],[128,2,1]]
# scale_args = [1.2,1.35]
# block_args = rescale_block(block_args, scale_args, 4)
# print(block_args) # [[96, 1, 2], [48, 1, 2], [128, 2, 3], [272, 2, 3]]


class Swish(nn.Module):
    def __init__(self, inplace=False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.mul_(x.sigmoid()) if self.inplace else x.mul(x.sigmoid())

class Attention_Layer(nn.Module):
    def __init__(self, out_channel, att_type='stja', act=Swish(inplace=True)):
        super(Attention_Layer, self).__init__()

        __attention = {
            'stja': ST_Joint_Att,
            'pa': Part_Att,
            'ca': Channel_Att,
            'fa': Frame_Att,
            'ja': Joint_Att,
        }

        self.att = __attention[att_type](channel=out_channel)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = act

    def forward(self, x):
        res = x
        x = x * self.att(x)
        return self.act(self.bn(x) + res)


class ST_Joint_Att(nn.Module):
    def __init__(self, channel, reduct_ratio=4, bias=True):
        super(ST_Joint_Att, self).__init__()

        inner_channel = channel // reduct_ratio

        self.fcn = nn.Sequential(
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(inner_channel),
            nn.Hardswish(),
        )
        self.conv_t = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.conv_v = nn.Conv2d(inner_channel, channel, kernel_size=1)

    def forward(self, x):
        # print('attention',x.shape)  # 2 48 288 25
        N, C, T, V = x.size()
        x_t = x.mean(3, keepdims=True)  # 2 48 288 1
        x_v = x.mean(2, keepdims=True).transpose(2, 3)  # 2 48 1 25 -> 2 48 25 1 
        x_att = self.fcn(torch.cat([x_t, x_v], dim=2))  # 2 48 288+25 1 > 2 48/reduce 288+25 1 
        x_t, x_v = torch.split(x_att, [T, V], dim=2)
        x_t_att = self.conv_t(x_t).sigmoid()  # 2 48 288 1 
        x_v_att = self.conv_v(x_v.transpose(2, 3)).sigmoid()
        x_att = x_t_att * x_v_att
        # print('x_att',x_att.shape)  # 2 48 288 25 
        return x_att


class Part_Att(nn.Module):
    def __init__(self, channel, parts, reduct_ratio, bias, **kwargs):
        super(Part_Att, self).__init__()

        self.parts = parts
        self.joints = nn.Parameter(self.get_corr_joints(), requires_grad=False)
        inner_channel = channel // reduct_ratio

        self.softmax = nn.Softmax(dim=3)
        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, channel*len(self.parts), kernel_size=1, bias=bias),
        )

    def forward(self, x):
        N, C, T, V = x.size()
        x_att = self.softmax(self.fcn(x).view(N, C, 1, len(self.parts)))
        x_att = x_att.index_select(3, self.joints).expand_as(x)
        return x_att

    def get_corr_joints(self):
        num_joints = sum([len(part) for part in self.parts])
        joints = [j for i in range(num_joints) for j in range(len(self.parts)) if i in self.parts[j]]
        return torch.LongTensor(joints)


class Channel_Att(nn.Module):
    def __init__(self, channel, **kwargs):
        super(Channel_Att, self).__init__()

        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel//4, kernel_size=1),
            nn.BatchNorm2d(channel//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//4, channel, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.fcn(x)


class Frame_Att(nn.Module):
    def __init__(self, **kwargs):
        super(Frame_Att, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv2d(2, 1, kernel_size=(9,1), padding=(4,0))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=2).transpose(1, 2)
        return self.conv(x)


class Joint_Att(nn.Module):
    def __init__(self, parts, **kwargs):
        super(Joint_Att, self).__init__()

        num_joint = sum([len(part) for part in parts])

        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_joint, num_joint//2, kernel_size=1),
            nn.BatchNorm2d(num_joint//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_joint//2, num_joint, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fcn(x.transpose(1, 3)).transpose(1, 3)


# Thanks to YAN Sijie for the released code on Github (https://github.com/yysijie/st-gcn)
class SpatialGraphConv(nn.Module):
    def __init__(self, in_channel, out_channel, max_graph_distance, A, bias=True, edge=True):
        super(SpatialGraphConv, self).__init__()

        self.s_kernel_size = max_graph_distance + 1
        self.gcn = nn.Conv2d(in_channel, out_channel*self.s_kernel_size, 1, bias=bias)
        self.A = nn.Parameter(A[:self.s_kernel_size], requires_grad=False)
        if edge:
            self.edge = nn.Parameter(torch.ones_like(self.A))
        else:
            self.edge = 1

    def forward(self, x):
        x = self.gcn(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, self.A * self.edge)).contiguous()
        return x

class Zero_Layer(nn.Module):
    def __init__(self):
        super(Zero_Layer, self).__init__()

    def forward(self, x):
        return 0



class Basic_Layer(nn.Module):
    def __init__(self, in_channel, out_channel, residual=True, bias=True, act=Swish(inplace=True)):
        super(Basic_Layer, self).__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel)

        self.residual = nn.Identity() if residual else Zero_Layer()
        self.act = act

    def forward(self, x):
        # print('basic layer x',x.shape) # 2 6 288 25 
        res = self.residual(x)
        x = self.act(self.bn(self.conv(x)) + res)
        # print('basic layer output', x.shape)  # 2 64 288 25
        return x


class Spatial_Graph_Layer(Basic_Layer):
    def __init__(self, in_channel, out_channel, max_graph_distance, A, bias=True, residual=True):
        super(Spatial_Graph_Layer, self).__init__(in_channel, out_channel, residual, bias)

        self.conv = SpatialGraphConv(in_channel, out_channel, max_graph_distance, A)
        if residual and in_channel != out_channel:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, bias=bias),
                nn.BatchNorm2d(out_channel),
            )


class Temporal_Basic_Layer(Basic_Layer):
    def __init__(self, channel, temporal_window_size, bias=True, stride=1, residual=True):
        super(Temporal_Basic_Layer, self).__init__(channel, channel, residual, bias)

        padding = (temporal_window_size - 1) // 2 
        self.conv = nn.Conv2d(channel, channel, (temporal_window_size, 1), (stride, 1), (padding, 0), bias=bias)
        if residual and stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(channel, channel, 1, (stride, 1), bias=bias),
                nn.BatchNorm2d(channel)
            )

class Temporal_Sep_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act=Swish(inplace=True), expand_ratio=2, stride=1, residual=True):
        super(Temporal_Sep_Layer, self).__init__()

        padding = (temporal_window_size - 1) // 2
        self.act = act

        if expand_ratio > 0:
            inner_channel = channel * expand_ratio
            self.expand_conv = nn.Sequential(
                nn.Conv2d(channel, inner_channel, 1, bias=bias),
                nn.BatchNorm2d(inner_channel),
            )
        else:
            inner_channel = channel
            self.expand_conv = None

        self.depth_conv = nn.Sequential(
            nn.Conv2d(inner_channel, inner_channel, (temporal_window_size,1), (stride,1), (padding,0), groups=inner_channel, bias=bias),
            nn.BatchNorm2d(inner_channel),
        )
        self.point_conv = nn.Sequential(
            nn.Conv2d(inner_channel, channel, 1, bias=bias),
            nn.BatchNorm2d(channel),
        )
        if not residual:
            self.residual = Zero_Layer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channel, channel, 1, (stride,1), bias=bias),
                nn.BatchNorm2d(channel),
            )

    def forward(self, x):
        res = self.residual(x)
        if self.expand_conv is not None:
            x = self.act(self.expand_conv(x))
        x = self.act(self.depth_conv(x))
        x = self.point_conv(x)
        return x + res


class EfficientGCN_Blocks(nn.Sequential):
    def __init__(self, init_channel, block_args, kernel_size, A, input_channel=0):
        super(EfficientGCN_Blocks, self).__init__()

        temporal_window_size, max_graph_distance = kernel_size 

        if input_channel > 0:
            self.add_module('init_bn', nn.BatchNorm2d(input_channel))
            self.add_module('stem_scn', Spatial_Graph_Layer(input_channel, init_channel, max_graph_distance, A=A))
            self.add_module('stem_tcn', Temporal_Basic_Layer(init_channel, temporal_window_size))
        
        last_channel = init_channel 
        temporal_layer = Temporal_Sep_Layer

        for i, [channel, stride, depth] in enumerate(block_args):
            self.add_module(f'block-{i}_scn', Spatial_Graph_Layer(last_channel, channel, max_graph_distance, A=A))
            for j in range(depth):
                s = stride if j == 0 else 1
                self.add_module(f'block-{i}_tcn-{j}', temporal_layer(channel, temporal_window_size, bias=True, stride=s))
            self.add_module(f'block-{i}_att', Attention_Layer(channel))
            last_channel = channel


@BACKBONES.register_module()
class EfficientGCN(nn.Module):

    def __init__(self, 
                data_shape=[3, 6, 300, 25, 2], 
                block_args=[[96, 1, 2], [48, 1, 2], [128, 2, 3], [272, 2, 3]], 
                fusion_stage=2, 
                stem_channel=64, 
                gragh_config=dict(layout='ntu-rgb+d', strategy='spatial')):
        super().__init__()

        self.graph = Graph(**gragh_config)
        A = torch.tensor(
            self.graph.A, dtype=torch.float32, requires_grad=False)

        num_input, num_channel, _, _, _ = data_shape  # branch channel frame joint person [3, 6, 300, 25, 2]

        # input branches 
        self.input_branches = nn.ModuleList([EfficientGCN_Blocks(
            init_channel = stem_channel,
            block_args = block_args[:fusion_stage],
            input_channel = num_channel,
            kernel_size=[5,2],
            A=A,
        ) for _ in range(num_input)])

        # main stream
        last_channel = stem_channel if fusion_stage == 0 else block_args[fusion_stage-1][0]
        self.main_stream = EfficientGCN_Blocks(
            init_channel = num_input * last_channel,
            block_args = block_args[fusion_stage:],
            kernel_size=[5,2],
            A=A,
        )

        # output
        # last_channel = num_input * block_args[-1][0] if fusion_stage == len(block_args) else block_args[-1][0]
        # self.classifier = nn.Sequential(
        #     nn.AdaptiveAvgPool3d(1),
        #     nn.Dropout(0.25, inplace=True),
        #     nn.Conv3d(last_channel, num_class, kernel_size=1)
        # )


    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        # if isinstance(self.pretrained, str):
        #     logger = get_root_logger()
        #     logger.info(f'load model from: {self.pretrained}')

        #     load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        # elif self.pretrained is None:
        #     for m in self.modules():
        #         if isinstance(m, nn.Conv2d):
        #             kaiming_init(m)
        #         elif isinstance(m, nn.Linear):
        #             normal_init(m)
        #         elif isinstance(m, _BatchNorm):
        #             constant_init(m, 1)
        # else:
        #     raise TypeError('pretrained must be a str or None')
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):
        """Defines the computation performed at every call.
        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        # data normalization
        x = x.float()
        # n, c, t, v, m = x.size()  # bs 3 300 25(17) 2

        n, n_stream, c, t, v, m = x.size() # bs 3  C T V M -- 2 3 6 300 25 2
        # print('input size--',x.size())
        x = x.permute(1, 0, 5, 2, 3, 4).contiguous().view(n_stream, n*m, c, t, v) 
        # print('permute --',x.shape)  # 3 bs*M C T V --- 3 4 6 300 25
 
        # input branches
        x = torch.cat([branch(x[i]) for i, branch in enumerate(self.input_branches)], dim=1)
        # print('input branches', x.shape) # 2 96 288 25   4 144 300 25


        # main stream 
        x = self.main_stream(x)
        # print('after main_stream', x.shape)  # 4 272 75 25

        # output 
        _, C, T, V = x.size()
        feature = x.view(n, m, C, T, V).permute(0, 2, 3, 4, 1)  # 2 2 272 75 25 -> 2 272 75 25 2
        
        return feature
        # out = self.classifier(feature).view(n, -1)
        # print('after classifier--', out.shape) # 2 60 
    
        # return out 
        


        # x = x.permute(0, 4, 3, 1, 2).contiguous()  # N M V C T
        # x = x.view(n * m, v * c, t)
        # x = self.data_bn(x)
        # x = x.view(n, m, v, c, t)
        # x = x.permute(0, 1, 3, 4, 2).contiguous()
        # x = x.view(n * m, c, t, v)  # bsx2 3 300 25(17)

        # # forward
        # for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
        #     x, _ = gcn(x, self.A * importance)

        # return x
