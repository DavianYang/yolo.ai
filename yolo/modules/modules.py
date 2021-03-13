from typing import Any, List

from torch import nn
import torch.nn.functional as F
from torch import Tensor

def make_layers(cfg: List[tuple], in_channels: int) -> nn.Sequential:
    layers = nn.ModuleList()

    for x in cfg:
        if "Conv" in str(type(x)):
            kernel_size, filters, stride, padding = x
            layers += [ConvBlock(in_channels, filters, kernel_size, stride, padding)]

            in_channels = filters

        elif "ConvWithoutBN" in str(type(x)):
            kernel_size, filters, stride, padding = x
            layers += [nn.Conv2d(in_channels, filters, kernel_size, stride, padding)]

            in_channels = filters

        elif "MaxPool" in str(type(x)):
            kernel_size, stride = x
            layers += [nn.MaxPool2d(kernel_size, stride)]

        elif "RepeatWithResidual" in str(type(x)):
            for _ in range(x.nums):
                layers += [ResBlock(in_channels, x.blocks)]
                in_channels = x.blocks[-1].filters

        elif "Repeat" in str(type(x)):
            for _ in range(x.nums):
                for conv in x.blocks:
                    filters, kernel_size, stride, padding = conv
                    layers += [ConvBlock(in_channels, conv.filters, kernel_size, stride, padding)]

                    in_channels = filters
        
        elif "ScalePred" in str(type(x)):
            num_classes, num_anchors = x
            layers += [ScalePrediction(in_channels, num_classes, num_anchors)]

    return nn.Sequential(*layers)

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        blocks: list
    ) -> None:
        super().__init__()
        conv1 = blocks[0]
        conv2 = blocks[1]
        self.resblock = nn.Sequential(
            ConvBlock(channels, conv1.filters, kernel_size=conv1.kernel_size, stride=conv1.stride, padding=conv1.padding),
            ConvBlock(conv1.filters, conv2.filters, kernel_size=conv2.kernel_size, stride=conv2.stride, padding=conv2.padding)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.resblock(x) + x
        return x

class Upsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.upsample = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size, stride, padding),
            nn.Upsample(scale_factor=2, mode="nearest")
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.upsample(x)
        

class ScalePrediction(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        num_anchors
    ) -> Any:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.pred = nn.Sequential(
            ConvBlock(in_channels, 2*in_channels, kernel_size=3, padding=1),
            nn.Conv2d(2*in_channels, (num_classes + 5) * num_anchors, kernel_size=1)
        )
        
    def forward(self, x: Tensor) -> Tensor:
        return (
            self.pred(x)
            .reshape(x.shape[0],  self.num_anchors, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2) # N x num_anchors x 13 x 13 x 5 + num_classes
        )

# Thanks to Zhenliang He for the code for SpaceToDepth
# https://discuss.pytorch.org/t/is-there-any-layer-like-tensorflows-space-to-depth-function/3487/15   
class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size
    
    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(
            N, C, H // self.bs, self.bs, W // self.bs, self.bs
        )
        
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous() # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs) # (N, C*bs^2, H//bs, W//bs)


class ReorgBlock(nn.Module):
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