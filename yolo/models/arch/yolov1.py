import torch
from torch import nn
from torch import Tensor

from yolo.models.backbones.darknet import darknet

class YOLOv1(nn.Module):
    def __init__(
        self,
        grid_size: int = 7,
        num_boxes: int = 2,
        num_classes: int = 20,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        
        self.backbone = darknet()
        self.fcs = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * self.grid_size * self.grid_size, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, self.grid_size * self.grid_size * (self.num_classes + self.num_boxes * 5))
        )
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        return self.fcs(torch.flatten(x, start_dim=1))