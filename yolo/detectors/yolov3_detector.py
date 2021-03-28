from typing import Tuple, List

import torch
from torch import nn
from torch import Tensor

from cfg.detectors.yolov3_detector_cfg import small_scale_cfg, medium_scale_cfg, large_scale_cfg
from yolo.modules.modules import ConvBlock, Upsample, ScalePrediction


class YOLOv3Detector(nn.Module):
    def __init__(self, num_classes: int, num_anchors: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        self.small_1 = self._make_layers(small_scale_cfg[0], 1024)
        self.small_b = self._make_layers(small_scale_cfg[1], 1024)
        self.small_2 = self._make_layers(small_scale_cfg[2], 512)
        self.upsample1 = Upsample(512, 256, kernel_size=1, stride=1)
        
        self.medium_1 = self._make_layers(medium_scale_cfg[0], 768)
        self.medium_b = self._make_layers(medium_scale_cfg[1], 512)
        self.medium_2 = self._make_layers(medium_scale_cfg[2], 256)
        self.upsample2 = Upsample(256, 128, kernel_size=1, stride=1)
        
        self.large_1 = self._make_layers(large_scale_cfg, 384)
    
    def forward(self, small: Tensor, medium: Tensor, large: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.small_1(small)
        small_branch = self.small_b(x)
        small_out = self.small_2(small_branch)
        x = self.upsample1(small_branch)
        
        x = torch.cat([x, medium], dim=1)
        x = self.medium_1(x)
        medium_branch = self.medium_b(x)
        medium_out = self.medium_2(medium_branch)
        x = self.upsample2(medium_branch)
        
        x = torch.cat([x, large], dim=1)
        large_out = self.large_1(x)
        
        return small_out, medium_out, large_out
    
    def _make_layers(self, cfg: List[tuple], in_channels: int) -> nn.Sequential:
        layers = nn.ModuleList()

        for x in cfg:
            if "Conv" in str(type(x)):
                kernel_size, filters, stride, padding = x
                layers += [ConvBlock(in_channels, filters, kernel_size, stride, padding)]

                in_channels = filters

            elif "Repeat" in str(type(x)):
                for _ in range(x.nums):
                    for conv in x.blocks:
                        filters, kernel_size, stride, padding = conv
                        layers += [ConvBlock(in_channels, conv.filters, kernel_size, stride, padding)]

                        in_channels = filters
            
            elif "ScalePred" in str(type(x)):
                layers += [ScalePrediction(in_channels, self.num_classes, self.num_anchors)]
                
        return nn.Sequential(*layers)