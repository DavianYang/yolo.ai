from typing import List, Tuple
import torch
from torch import nn

from yolo.losses.base import YOLOLoss

class YOLOv2Loss(nn.Module):
    def __init__(
        self,
        anchor_boxes: List[Tuple[int]],
        image_size: int = 416,
        grid_size: int = 13,
    ):
        super().__init__()
        self.anchor_boxes = torch.tensor(anchor_boxes)
        self.loss = YOLOLoss(self.anchor_boxes[0] / (image_size / grid_size))
        
    def forward(self, pred, target):
        loss = self.loss(pred, target)
        return loss
        