from typing import IO
import torch
from torch import nn
from torch.nn import functional as F

from yolo.utils.utils import IOU

class YOLOLoss(nn.Module):
    def __init__(self, anchor_boxes: list, S: int, device: str):
        super().__init__()
        self.anchor_boxes = anchor_boxes
        self.mse = nn.MSELoss()
        self.anchor_boxes = anchor_boxes
        self.S = S
        self.device = device
        
    def forward(self, pred, target):
        exist_mask = target[..., 4:5]
        existing_boxes = exist_mask * pred
        cell_idx = torch.arange(self.S, device=self.device)
        bx = exist_mask * torch.sigmoid(
            pred[..., 0:1]
        ) + exist_mask * cell_idx.view([1, 1, -1, 1, 1])
        by = exist_mask * torch.sigmoid(
            pred[..., 1:2]
        ) + exist_mask * cell_idx.view([1, -1, 1, 1, 1])
        bw = (
            exist_mask
            * self.anchor_boxes[:, 2].view([1, 1, 1, -1, 1])
            * exist_mask
            * torch.exp(pred[..., 3:4])
        )
        bh = (
            exist_mask
            * self.anchor_boxes[:, 3].view([1, 1, 1, -1, 1])
            * exist_mask
            * torch.exp(pred[..., 3:4])
        )
        
        ious = IOU(
            torch.cat([bx, by, bw, bh], dim=-1), target[..., :4]
        )

        xy_loss = self.mse(torch.cat([bx, by], dim=-1), target[..., :2])
        bwbh = torch.cat([bw, bh], dim=-1)
        wh_loss = self.mse(
            torch.sqrt(torch.abs(bwbh) + 1e-32),
            torch.sqrt(torch.abs(target[..., 2:4]) + 1e-32),
        )
        obj_loss = self.mse(
            exist_mask, exist_mask * ious * torch.sigmoid(existing_boxes[..., 4:5])
        )
        no_obj_loss = self.mse(
            (1 - exist_mask),
            (
                ((1 - exist_mask) * (1 - torch.sigmoid(pred[..., 4:5])))
                * ((ious.max(-1)[0] < 0.6).int().unsqueeze(-1))
            ),
        )
        class_loss = F.nll_loss(
            (exist_mask * F.log_softmax(pred[..., 5:], dim=-1)).flatten(
                end_dim=-2
            ),
            target[..., 5:].flatten(end_dim=-2).argmax(-1),
        )
        return 5 * xy_loss + 5 * wh_loss + obj_loss + no_obj_loss + class_loss
    

class YOLOv3Loss(nn.Module):
    def __init__(self, anchor_boxes: list, device: str):
        super().__init__()
        self.small_loss = YOLOLoss(anchor_boxes[2] / (416 / 13), S = 13, device=device)
        self.medium_loss = YOLOLoss(anchor_boxes[1] / (416 / 26), S = 26, device=device)
        self.large_loss = YOLOLoss(anchor_boxes[0] / (416 / 52), S = 52, device=device)
    
    def forward(self, pred, target, device):
        s_loss = self.small_loss(pred[0], target[0], device)
        m_loss = self.medium_loss(pred[1], target[1], device)
        l_loss = self.large_loss(pred[2], target[2], device)
        
        return s_loss + m_loss + l_loss
        

# class YOLOLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mse = nn.MSELoss()
#         self.bce = nn.BCEWithLogitsLoss()
#         self.entropy = nn.CrossEntropyLoss()
#         self.sigmoid = nn.Sigmoid()
        
#         # Constants
#         self.lambda_class = 1
#         self.lambda_noobj = 10
#         self.lambda_obj = 1
#         self.lambda_box = 10
        
    
#     def forward(self, predictions, target, anchors):
#         obj = target[..., 0] == 1
#         noobj = target[..., 0] == 0
        
#         # No Object Loss
#         no_object_loss = self.bce(
#             (predictions[..., 0:1][noobj], (target[..., 0:1][noobj]))
#         )
        
#         # Object Loss
#         anchors = anchors.reshape(1, 3, 1, 1, 2)
#         # bx, by = sigmoid(x, y), w, h = p_w * exp(t_w)
#         box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5] * anchors)], dim=-1)
#         iou = IOU(box_preds[obj], target[..., 1:5][obj]).detach()
#         object_loss = self.bce((predictions[..., 0:1][obj]), (iou * target[..., 0:1]))
        
#         # Box Coordinate Loss
#         predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])
#         target[...,3:5] = torch.log(
#             (1e-16 + target[..., 3:5] / anchors)
#         )
#         box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])
        
#         # Class Loss
#         class_loss = self.entropy(
#             (predictions[..., 5:][obj]), (target[..., 5][obj].long())
#         )
        
#         return (
#             self.lambda_box * box_loss
#             + self.lambda_obj * object_loss
#             + self.lambda_noobj * no_object_loss
#             + self.lambda_class * class_loss
#         )