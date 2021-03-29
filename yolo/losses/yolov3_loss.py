import torch
from torch import nn

from yolo.losses.base import YOLOLoss

class YOLOv3Loss(nn.Module):
    def __init__(
        self,
        anchor_boxes: list, 
        device: str,
        image_size: int = 416,
        grid_size: list = [52, 26, 13], 
    ):
        super().__init__()
        self.anchor_boxes = torch.tensor(anchor_boxes)
        
        self.small_loss = YOLOLoss(self.anchor_boxes[2] / (image_size / grid_size[2]),
                                   grid_size=grid_size[2],
                                   device=device)
        self.medium_loss = YOLOLoss(self.anchor_boxes[1] / (image_size / grid_size[1]),
                                    grid_size=grid_size[1],
                                    device=device)
        self.large_loss = YOLOLoss(self.anchor_boxes[0] / (image_size / grid_size[0]),
                                   grid_size=grid_size[0], 
                                   device=device)
    
    def forward(self, pred, target):
        s_loss = self.small_loss(pred[0], target[0])
        m_loss = self.medium_loss(pred[1], target[1])
        l_loss = self.large_loss(pred[2], target[2])
        
        return s_loss + m_loss + l_loss