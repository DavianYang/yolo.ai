from typing import List, Tuple

import torch
from torch import nn

from yolo.models.metrics.functional import IOU

class YOLOLoss(nn.Module):
    def __init__(self, anchors_boxes: List[Tuple[float]]):
        super().__init__()
        self.anchor_boxes = anchors_boxes.reshape(1, 3, 1, 1, 2)
        
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, pred, target):
        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0

        return (
            self.lambda_obj * self._obj_loss_fn(pred, target, obj)
            + self.lambda_noobj * self._no_obj_loss_fn(pred, target, noobj)
            + self.lambda_box * self._box_coord_loss_fn(pred, target, obj)
            + self.lambda_class * self._class_loss_fn(pred, target, obj)
        )
        
    def _no_obj_loss_fn(self, pred, target, noobj):
        return self.bce(
            (pred[..., 0:1][noobj]), (target[..., 0:1][noobj]),
        )
    
    def _obj_loss_fn(self, pred, target, obj):
        box_preds = torch.cat([self.sigmoid(pred[..., 1:3]), torch.exp(pred[..., 3:5]) * self.anchor_boxes], dim=-1)
        ious = IOU(box_preds[obj], target[..., 1:5][obj]).detach()
        return self.mse(self.sigmoid(pred[..., 0:1][obj]), ious * target[..., 0:1][obj])
    
    def _box_coord_loss_fn(self, pred, target, obj):
        pred[..., 1:3] = self.sigmoid(pred[..., 1:3])  # x,y coordinates
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / self.anchor_boxes)
        )  # width, height coordinates
        return self.mse(pred[..., 1:5][obj], target[..., 1:5][obj])
    
    def _class_loss_fn(self, pred, target, obj):
        return self.entropy(
            (pred[..., 5:][obj]), (target[..., 5][obj].long()),
        )