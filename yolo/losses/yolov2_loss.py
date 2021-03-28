import torch
from torch import nn

from yolo.losses.base import YOLOLoss

class YOLOv2Loss(nn.Module):
    def __init__(
        self,
        anchor_boxes: list, 
        device: str,
        image_size: int = 416,
        grid_size: list = [13],
    ):
        super().__init__()
        self.anchor_boxes = torch.tensor(anchor_boxes)
        self.grid_size = grid_size
        
        self.loss = YOLOLoss(self.anchor_boxes[2] / (image_size / grid_size[2]),
                                   grid_size=grid_size[2],
                                   device=device)
        
    def forward(self, pred, target):
        loss = self.loss(pred, target)
        return loss
        