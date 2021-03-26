import math

from torch import nn
from torch import Tensor

from cfg.backbones.efficientnet import compound_params_dict
from yolo.modules.modules import ConvBlock, MBConvBlock

class EfficientNet(nn.Module):
    def __init__(self, version: str, num_classes: int, base_model: list) -> None:
        super().__init__()
        width_factor, depth_factor, drop_rate = self.calculate_factors(version)
        last_channels = math.ceil(1280 * width_factor)
        self.base_model = base_model
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature = self.create_features(width_factor, depth_factor, last_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(last_channels, num_classes)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(self.feature(x))
        return self.classifier(x.view(x.shape[0], -1))
       
    def calculate_factors(self, version: str, alpha: float=1.2, beta: float=1.1):
        phi, _, drop_rate = compound_params_dict[version]
        depth_factor = alpha ** phi
        width_factor = beta ** phi
        return width_factor, depth_factor, drop_rate
    
    def create_features(self, width_factor: float, depth_factor: float, last_channels: int):
        channels = int(32 * width_factor)
        features = [ConvBlock(3, channels, kernel_size=3, stride=2, padding=1, activation='silu')]
        in_channels = channels
        
        for expand_ratio, repeats, kernel_size, channels, stride in self.base_model:
            out_channels = 4 * math.ceil(int(channels * width_factor) / 4)
            layers_repeat = math.ceil(repeats * depth_factor)
            
            for layer in range(layers_repeat):
                features.append(
                    MBConvBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride = stride if layer == 0 else 1,
                        padding=kernel_size // 2, # if k=1:pad=0, k=3:pad=1 k=5:pad=2
                        expand_ratio=expand_ratio
                    )
                )
                in_channels = out_channels
                
        features.append(ConvBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=0, activation='silu'))
        return nn.Sequential(*features)