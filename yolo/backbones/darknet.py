import torch
import torch.nn as nn

from yolo.modules.modules import ConvBlock, ResBlock

darknet_cfg = [
    # Tuple: (kernel_size, num_filters, stride, padding)
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    # List: 
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


darknet19_cfg = {
    'conv0': [32],
    'conv1': ['M', 64],
    'conv2': ['M', 128, 64, 128],
    'conv3': ['M', 256, 128, 256],
    'conv4': ['M', 512, 256, 512, 256, 512],
    'conv5': ['M', 1024, 512, 1024, 512, 1024]
}

class DarkNet(nn.Module):
    def __init__(
        self,
        cfg: dict = darknet_cfg,
        in_channels: int = 3
    ):
        super().__init__()
        self.cfg = cfg
        self.in_channels = in_channels
        self.features = self._make_layers(cfg)
        
    def forward(self, x):
        return self.features(x)

        
    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_channels
        
        for x in cfg:
            if type(x) == tuple:
                layers += [ConvBlock(in_channels, x[1], kernel=x[0], stride=x[2], padding=x[3])]
                
                in_channels = x[1]
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repats = x[2]
                
                for _ in range(num_repats):
                    layers += [ConvBlock(in_channels, conv1[1], kernel=conv1[0], stride=conv1[2], padding=conv1[3])]
                    layers += [ConvBlock(conv1[1], conv2[1], kernel=conv2[0], stride=conv2[2], padding=conv2[3])]
                    
                    in_channels = conv2[1]
        return nn.Sequential(*layers)
            


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