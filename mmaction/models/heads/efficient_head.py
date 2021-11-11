import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead


@HEADS.register_module()
class EfficientHead(BaseHead):
    """The classification head for STGCN.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        num_person (int): Number of person. Default: 2.
        init_std (float): Std value for Initiation. Default: 0.01.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 num_person=2,
                 init_std=0.01,
                 drop_out=0.25):
        super().__init__(num_classes, in_channels, loss_cls)

        # self.spatial_type = spatial_type
        # self.in_channels = in_channels
        # self.num_classes = num_classes
        # self.num_person = num_person
        # self.init_std = init_std

        # self.pool = None
        # if self.spatial_type == 'avg':
        #     self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # elif self.spatial_type == 'max':
        #     self.pool = nn.AdaptiveMaxPool2d((1, 1))
        # else:
        #     raise NotImplementedError

        # self.fc = nn.Conv2d(self.in_channels, self.num_classes, kernel_size=1)

        # if drop_out:
        #     self.drop_out = nn.Dropout(drop_out)
        # else:
        #     self.drop_out = lambda x: x
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Dropout(drop_out, inplace=True),
            nn.Conv3d(in_channels, num_classes, kernel_size=1)
        )

    def init_weights(self):
        # normal_init(self.fc, std=self.init_std)
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
        # # global pooling
        # assert self.pool is not None
        # # print('before pool--', x.shape) # 32 256 75 17
        # x = self.pool(x)
        # # print('after pool--',x.shape)  # 32 256 1 1 
        # x = x.view(x.shape[0] // self.num_person, self.num_person, -1, 1,
        #            1).mean(dim=1)
        # # print('before fc --', x.shape) # bs 256 1 1 
        
        # x = self.drop_out(x)
        # # prediction
        # x = self.fc(x)
        # x = x.view(x.shape[0], -1)
        # # print('stgcn-head',x.shape)
        N, C, T, V, M = x.shape
        x = self.classifier(x).view(N, -1)

        return x
