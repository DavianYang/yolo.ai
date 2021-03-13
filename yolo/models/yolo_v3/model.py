from typing import Tuple

import torch
from torch import nn
from torch import Tensor

from yolo.modules.modules import Upsample, make_layers
from yolo.backbones.darknet import darknet53
from yolo.config.darknet_cfg import small_scale_cfg, medium_scale_cfg, large_scale_cfg

class MultiScaleDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.small_1 = make_layers(small_scale_cfg[0], 1024)
        self.small_b = make_layers(small_scale_cfg[1], 1024)
        self.small_2 = make_layers(small_scale_cfg[2], 512)
        self.upsample1 = Upsample(512, 256, kernel_size=1, stride=1)
        
        self.medium_1 = make_layers(medium_scale_cfg[0], 768)
        self.medium_b = make_layers(medium_scale_cfg[1], 512)
        self.medium_2 = make_layers(medium_scale_cfg[2], 256)
        self.upsample2 = Upsample(256, 128, kernel_size=1, stride=1)
        
        self.large_1 = make_layers(large_scale_cfg, 384)
    
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
    

class YOLOv3(nn.Module):
    def __init__(
        self, 
        anchor_boxes: list,
        num_anchors: int = 3
    ) -> None:
        super().__init__()
        self.anchor_boxes = torch.tensor(anchor_boxes)
        self.num_anchors = num_anchors
        
        self.base_model = darknet53()
        self.detector = MultiScaleDetector()
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        s, m, l = self.base_model(x)
        s, m, l = self.detector(s, m, l)
        return s, m, l
        