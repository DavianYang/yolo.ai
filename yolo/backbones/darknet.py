from typing import TypedDict

import torch
import torch.nn as nn

from yolo.modules.modules import ConvBlock, ResBlock

darknet19_cfg = {
    'conv0': [32],
    'conv1': ['M', 64],
    'conv2': ['M', 128, 64, 128],
    'conv3': ['M', 256, 128, 256],
    'conv4': ['M', 512, 256, 512, 256, 512],
    'conv5': ['M', 1024, 512, 1024, 512, 1024]
}

class DarkNet19(nn.Module):
    def __init__(
        self,
        cfg: dict = darknet19_cfg,
        in_channels: int = 3,
        ):
        super().__init__()
        self.cfg = cfg
        self.in_channels = in_channels

        self.conv0 = self._make_layers(self.cfg['conv0'])
        self.conv1 = self._make_layers(self.cfg['conv1'])
        self.conv2 = self._make_layers(self.cfg['conv2'])
        self.conv3 = self._make_layers(self.cfg['conv3'])
        self.conv4 = self._make_layers(self.cfg['conv4'])
        self.conv5 = self._make_layers(self.cfg['conv5'])

    
    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        
        c_3 = self.conv3(x)
        c_4 = self.conv4(c_3)
        c_5 = self.conv5(c_4)
        
        return c_3, c_4, c_5
    
    
    def _make_layers(self, cfg):
        layers = []
        kernel = 3
        for x in cfg:
            padding = int(kernel / 3)
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [ConvBlock(self.in_channels, x, kernel, padding=padding)]
                kernel = 1 if kernel == 3 else 3
                self.in_channels = x
        return nn.Sequential(*layers)