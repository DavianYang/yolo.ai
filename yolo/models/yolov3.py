from typing import Tuple

import torch
from torch import nn
from torch import Tensor

from yolo.backbones.darknet import darknet53
from yolo.detectors.yolov3_detectors import Yolov3Detector
    
class YOLOv3(nn.Module):
    def __init__(
        self, 
        anchor_boxes: list,
        num_classes: int = 20,
        num_anchors: int = 3
    ) -> None:
        super().__init__()
        self.anchor_boxes = torch.tensor(anchor_boxes)
        self.num_anchors = num_anchors
        
        self.base_model = darknet53()
        self.detector = Yolov3Detector(num_classes, num_classes)
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        s, m, l = self.base_model(x)
        s, m, l = self.detector(s, m, l)
        return s, m, l
        