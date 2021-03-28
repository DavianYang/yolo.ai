import torch
import torch.nn as nn
from torch import Tensor

from yolo.backbones.darknet import darknet19
from yolo.modules.modules import ConvBlock, SpaceToDepth

class YOLOv2(nn.Module):
    def __init__(
        self,
        anchor_boxes: list,
        grid_size: int = 13,
        num_classes: int = 20,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.num_anchors = len(anchor_boxes)
        self.anchor_boxes = torch.tensor(anchor_boxes)
        
        # Detection Head
        self.darknet_head = darknet19(mode="head")
        self.middle = ConvBlock(512, 64, kernel_size=1, stride=1)
        self.space_to_depth = SpaceToDepth(block_size=2)
        self.darknet_tail = darknet19(mode="tail")
        
        self.conv = ConvBlock(1280, 1024, kernel_size=3, stride=1, padding=1)
        self.pred = nn.Conv2d(1024, self.anchor_boxes * (4 + 1 + num_classes), 
                               kernel_size=1, stride=1, bias=True)
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.darknet_head(x)
        middle = self.middle(x)
        middle = self.space_to_depth(middle)
        x = self.darknet_tail(x)
        x = torch.cat([middle, x], dim=1)
        
        x = self.conv(x)
        x = self.pred(x)
        
        x = x.permute(0, 2, 3, 1)
        x = x.view(-1, self.grid_size, self.grid_size, self.num_anchors, 4 + 1 + self.num_classes)
        return x
    
