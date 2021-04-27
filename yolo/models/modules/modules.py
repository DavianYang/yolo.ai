from typing import Sequence, Union, Callable, AnyStr, Any

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

from yolo.models.modules.activation import get_activation_layer

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence],
        stride: Union[int, Sequence] = 1,    
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        activation: Union[Callable, AnyStr] = (lambda: nn.ReLU(inplace=True))
    ) -> None:
        super().__init__()
        self.activate = (activation is not None)
        self.conv = nn.Conv2d(in_channels, out_channels, 
                      kernel_size, stride, 
                      padding, dilation,
                      groups, bias
                     )
        self.bn = nn.BatchNorm2d(out_channels)
        if self.activate:
            self.act = get_activation_layer(activation)
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.bn(self.conv(x))
        if self.activate:
            x = self.act(x)
        return x


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
            ConvBlock(channels, conv1.filters, 
                      kernel_size=conv1.kernel_size, 
                      stride=conv1.stride, padding=conv1.padding),
            ConvBlock(conv1.filters, conv2.filters, 
                      kernel_size=conv2.kernel_size, 
                      stride=conv2.stride, padding=conv2.padding)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.resblock(x) + x
        return x

class Upsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence],
        stride: Union[int, Sequence] = 1,
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
        
class SEBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        squeeze_channels: int, 
        activation: Union[Callable, AnyStr] = (lambda: nn.SiLU())
    ) -> None:
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, squeeze_channels, kernel_size=1),
            get_activation_layer(activation),
            nn.Conv2d(squeeze_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x: Tensor) -> Tensor:
        return x * self.se(x)
    
class MBConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence],
        stride: Union[int, Sequence],
        padding: int,
        expand_ratio: float,
        reduction: int = 4, # squeeze excitation
        survival_prob: float = 0.8 # for stochastic depth
    ) -> None:
        super().__init__()
        self.survival_prob = survival_prob
        self.use_residual = in_channels == out_channels and stride == 1
        
        hidden_dim = in_channels * expand_ratio
        
        self.expand = in_channels != hidden_dim
        squeeze_dim = int(in_channels / reduction)
        
        if self.expand:
            self.expand_conv = ConvBlock(
                in_channels, 
                hidden_dim, 
                kernel_size=3, 
                stride=1, 
                padding=1, 
                activation='silu'
            )
        self.conv = nn.Sequential(
            ConvBlock(
                hidden_dim, 
                hidden_dim, 
                kernel_size,
                stride,
                padding,
                groups=hidden_dim,
                activation='silu'
            ),
            SEBlock(hidden_dim, squeeze_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, inputs: Tensor) -> Tensor:
        x = self.expand_conv(inputs) if self.expand else inputs
        x = self.stochastic_depth(self.conv(x)) + inputs if self.use_residual else self.conv(x)
        return x
        
        
    def stochastic_depth(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        
        return torch.div(x, self.survival_prob) * binary_tensor