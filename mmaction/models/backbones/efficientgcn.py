import torch
import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init, normal_init
from mmcv.runner import load_checkpoint
from mmcv.utils import _BatchNorm

from ...utils import get_root_logger
from ..builder import BACKBONES
from ..skeleton_gcn.utils import Graph


class Attention_Layer(nn.Module):
    def __init__(self, out_channel, att_type, act, **kwargs):
        super(Attention_Layer, self).__init__()

        __attention = {
            'stja': ST_Joint_Att,
            'pa': Part_Att,
            'ca': Channel_Att,
            'fa': Frame_Att,
            'ja': Joint_Att,
        }

        self.att = __attention[att_type](channel=out_channel, **kwargs)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = act

    def forward(self, x):
        res = x
        x = x * self.att(x)
        return self.act(self.bn(x) + res)


class ST_Joint_Att(nn.Module):
    def __init__(self, channel, reduct_ratio, bias, **kwargs):
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
    def __init__(self, in_channel, out_channel, max_graph_distance, bias, edge, A, **kwargs):
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



class Basic_Layer(nn.Module):
    def __init__(self, in_channel, out_channel, residual, bias, act, **kwargs):
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
    def __init__(self, in_channel, out_channel, max_graph_distance, bias, residual=True, **kwargs):
        super(Spatial_Graph_Layer, self).__init__(in_channel, out_channel, residual, bias, **kwargs)

        self.conv = SpatialGraphConv(in_channel, out_channel, max_graph_distance, bias, **kwargs)
        if residual and in_channel != out_channel:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, bias=bias),
                nn.BatchNorm2d(out_channel),
            )


class Temporal_Basic_Layer(Basic_Layer):
    def __init__(slef, channel, temporal_window_size, bias, stride=1, residual=True, **kwargs):
        super(Temporal_Basic_Layer, self).__init__(channel, channel, residual, bias, **kwargs)

        padding = (temporal_window_size - 1) // 2 
        self.conv = nn.Conv2d(channel, channel, (temporal_window_size, 1), (stride, 1), (padding, 0), bias=bias)
        if residual and stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(channel, channel, 1, (stride, 1), bias=bias),
                nn.BatchNorm2d(channel)
            )

class Temporal_Sep_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, expand_ratio, stride=1, residual=True, **kwargs):
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
    def __init__(self, init_channel, block_args, layer_type, kernel_size, input_channel=0, **kwargs):
        super(EfficientGCN_Blocks, self).__init__()

        temporal_window_size, max_graph_distance = kenerl_size 

        if input_channel > 0:
            self.add_module('init_bn', nn.BatchNorm2d(input_channel))
            self.add_module('stem_scn', Spatial_Graph_Layer(input_channel, init_channel, max_graph_distance, **kwargs))
            self.add_module('stem_tcn', Temporal_Basic_Layer(init_channel, temporal_window_size, **kwargs))
        
        last_channel = init_channel 
        temporal_layer = Temporal_Sep_Layer

        for i, [channel, stride, depth] in enumerate(block_args):
            self.add_module(f'block-{i}_scn', Spatial_Graph_Layer(last_channel, channel, max_graph_distance, **kwargs))
            for j in range(depth):
                s = stride if j == 0 else 1
                self.add_module(f'block-{i}_tcn-{j}', temporal_layer(channel, temporal_window_size, stride=s, **kwargs))
            self.add_module(f'block-{i}_att', Attention_Layer(channel, **kwargs))
            last_channel = channel


def zero(x):
    """return zero."""
    return 0


def identity(x):
    """return input itself."""
    return x


class STGCNBlock(nn.Module):
    """Applies a spatial temporal graph convolution over an input graph
    sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and
            graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)`
            format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out},
            V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V,
            V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]
                `,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1),
                      (stride, 1), padding), nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True))

        if not residual:
            self.residual = zero

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = identity

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)), nn.BatchNorm2d(out_channels))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, adj_mat):
        """Defines the computation performed at every call."""
        res = self.residual(x)
        x, adj_mat = self.gcn(x, adj_mat)
        x = self.tcn(x) + res

        return self.relu(x), adj_mat


class ConvTemporalGraphical(nn.Module):
    """The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution.
            Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides
            of the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)`
            format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}
            , V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)
            ` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]
                `,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, adj_mat):
        """Defines the computation performed at every call."""
        assert adj_mat.size(0) == self.kernel_size

        # print('adj_mat', adj_mat.shape) # 3 17 17
        # print('x',x.shape)

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, adj_mat))

        return x.contiguous(), adj_mat


@BACKBONES.register_module()
class EfficientGCN(nn.Module):
    """Backbone of Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data.
        graph_cfg (dict): The arguments for building the graph.
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph. Default: True.
        data_bn (bool): If 'True', adds data normalization to the inputs.
            Default: True.
        pretrained (str | None): Name of pretrained model.
        **kwargs (optional): Other parameters for graph convolution units.

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, 
                data_shape, 
                block_args, 
                fusion_stage, 
                stem_channel, 
                **kwargs):
        super().__init__()

        num_input, num_channel, _, _, _ = data_shape  # branch channel frame joint person [3, 6, 300, 25, 2]

        # input branches 
        self.input_branches = nn.ModuleList([EfficientGCN_Blocks(
            init_channel = stem_channel,
            block_args = block_args[:fusion_stage],
            input_channel = num_channel,
            **kwargs
        ) for _ in range(num_input)])

        # main stream
        last_channel = stem_channel if fusion_stage == 0 else block_args[fusion_stage-1][0]
        self.main_stream = EfficientGCN_Blocks(
            init_channel = num_input * last_channel,
            block_args = block_args[fusion_stage:],
            **kwargs
        )

        # load graph
        # self.graph = Graph(**graph_cfg)
        # A = torch.tensor(
        #     self.graph.A, dtype=torch.float32, requires_grad=False)
        # self.register_buffer('A', A)

        # # build networks
        # spatial_kernel_size = A.size(0)
        # temporal_kernel_size = 9
        # kernel_size = (temporal_kernel_size, spatial_kernel_size)
        # self.data_bn = nn.BatchNorm1d(in_channels *
        #                               A.size(1)) if data_bn else identity

        # kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        # self.st_gcn_networks = nn.ModuleList((
        #     STGCNBlock(
        #         in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
        #     STGCNBlock(64, 64, kernel_size, 1, **kwargs),
        #     STGCNBlock(64, 64, kernel_size, 1, **kwargs),
        #     STGCNBlock(64, 64, kernel_size, 1, **kwargs),
        #     STGCNBlock(64, 128, kernel_size, 2, **kwargs),
        #     STGCNBlock(128, 128, kernel_size, 1, **kwargs),
        #     STGCNBlock(128, 128, kernel_size, 1, **kwargs),
        #     STGCNBlock(128, 256, kernel_size, 2, **kwargs),
        #     STGCNBlock(256, 256, kernel_size, 1, **kwargs),
        #     STGCNBlock(256, 256, kernel_size, 1, **kwargs),
        # ))

        # # initialize parameters for edge importance weighting
        # if edge_importance_weighting:
        #     self.edge_importance = nn.ParameterList([
        #         nn.Parameter(torch.ones(self.A.size()))
        #         for i in self.st_gcn_networks
        #     ])
        # else:
        #     self.edge_importance = [1 for _ in self.st_gcn_networks]

        # self.pretrained = pretrained



    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.Linear):
                    normal_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

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

        n, n_stream, c, t, v, m = x.size() # bs 3  C T V M 
        x = x.permute(1, 0, 5, 2, 3, 4).contiguous().view(n_stream, n*m, c, t, v)
 
        # input branches
        x = torch.cat([branch(x[i]) for i, branch in enumerate(self.input_branches)], dim=1)
        # print('input branches', x.shape) # 2 96 288 25

        


        x = x.permute(0, 4, 3, 1, 2).contiguous()  # N M V C T
        x = x.view(n * m, v * c, t)
        x = self.data_bn(x)
        x = x.view(n, m, v, c, t)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(n * m, c, t, v)  # bsx2 3 300 25(17)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        return x
