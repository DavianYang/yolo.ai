import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding, dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        blocks: int=1
    ):
        super().__init__()
        self.module_list = nn.ModuleList()
        for _ in range(blocks):
            resblock = nn.Sequential(
                ConvBlock(channels, channels // 2, kernel=1),
                ConvBlock(channels // 2, channels, kernel=3, padding=1)
            )
            self.module_list.append(resblock)
    
    
    def forward(self, x):
        for module in self.module_list:
            x = module(x) + x
        return x
    
    
class ReorgLayer(nn.Module):
    def __init__(
        self,
        stride: int
    ):
        """ To combine middle-level features and high-level features and getter classification accuracy """
        
        super().__init__()
        self.stride = stride
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        _height, _width = height // self.stride, width // self.stride
        
        x = x.view(batch_size, channels, _height, self.stride, _width, self.stride).transpose(3, 4).contiguous()
        x = x.view(batch_size, channels, _height * _width, self.stride * self.stride).transpose(2, 3).contiguous()
        x = x.view(batch_size, channels, self.stride * self.stride, _height * _width).transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, _height, _width)
        
        return x