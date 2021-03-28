from typing import Tuple

import torch
from torch import nn
from torch import Tensor

from yolo.backbones.darknet import darknet53
from yolo.detectors.yolov3_detector import YOLOv3Detector
    
class YOLOv3(nn.Module):
    def __init__(
        self, 
        anchor_boxes: list,
        num_classes: int = 20
    ) -> None:
        super().__init__()
        self.num_anchors = len(anchor_boxes)
        self.anchor_boxes = torch.tensor(anchor_boxes)
        
        self.backbone = darknet53()
        self.detector = YOLOv3Detector(num_classes, self.num_anchors)
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        s, m, l = self.backbone(x)
        s, m, l = self.detector(s, m, l)
        return s, m, l
        