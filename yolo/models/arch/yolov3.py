from typing import Tuple, List

import torch
from torch import nn
from torch import Tensor

from yolo.models.backbones.darknet import darknet53
from yolo.models.detectors.yolov3_detector import YOLOv3Detector
    
class YOLOv3(nn.Module):
    def __init__(
        self,
        num_classes: int = 20,
        num_anchors: int = 3
    ) -> None:
        super().__init__()
        self.num_anchors = num_anchors
        
        self.base_model = darknet53()
        self.detector = YOLOv3Detector(num_classes, num_classes)
        
    def forward(self, x: Tensor) -> Tuple[Tensor]:
        s, m, l = self.base_model(x)
        s, m, l = self.detector(s, m, l)
        return s, m, l
        