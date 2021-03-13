import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as numpy

from yolo.backbones.darknet import darknet19
from yolo.modules.modules import ConvBlock, SpaceToDepth


device = "cuda" if torch.cuda.is_available else "cpu"

class YOLOv2(nn.Module):
    def __init__(
        self,
        split_size: int = 13
        num_boxes: int = 5
        num_classes: int = 20,
        trainable: bool = False,
        anchor_boxes: list,
        device: str
    ):
        super().__init__()
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes
        
        # Detection Head
        self.darknet_head = darknet19(mode="head")
        self.middle = ConvBlock(512, 64, kernel=1, stride=1, padding=0)
        self.space_to_depth = SpaceToDepth(block_size=2)
        self.darknet_tail = darknet19(mode="tail")
        self.conv1 = ConvBlock(1280, 1024, kernel=3, stride=1, pad=1)
        self.pred = nn.Conv2d(1024, num_boxes * (4 + 1 + num_classes), 
                               kernel_size=1, stride=1, 
                               padding=0, bias=True)
        
        
        self.anchor_boxes = torch.tensor(anchor_boxes, device=self.device)
        self.num_anchors = len(anchor_boxes)
        self.trainable = trainable
        
        self.device = device
        
        self.grid_size = 0
        self.stride = 0.0
        
    def forward(self, x):
        x = self.darknet_head(x)
        middle = self.middle(x)
        middle = self.space_to_depth(middle)
        x = self.darknet_tail(x)
        x = torch.cat([middle, x], dim=1)
        x = self.conv1(x)
        x = self.pred(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(-1, self.S, self.S, self.B, 4 + 1 + self.C)
        return x
        
    def create_grid(self, grid_size):
        w, h = grid_size[1], grid_size[0]
        
        if self.grid_size == grid_size:
            # raise Error
            pass
            
        self.grid_size = grid_size
        self.stride = self.img_size / grid_size
        
        ws, hs = w // self.stride, h // self.stride
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        
        anchor_wh = self.anchors.repeat(hs*ws, 1, 1).unsqueeze(0).to(device)
        
        return grid_xy, anchor_wh
    
    def decode_boxes(self, xy, wh, requires_grad=False):
        xy = torch.add(xy, self.grid_size) 
    
