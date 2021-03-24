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
        self.anchor_boxes = anchor_boxes.to(device)
        self.S = S
        self.device = device
        
    def forward(self, pred, target):
        pred = pred.permute(0, 2, 3, 1, 4)
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
        ).to(self.device)
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
        self.anchor_boxes = torch.tensor(anchor_boxes)
        self.small_loss = YOLOLoss(self.anchor_boxes[2] / (416 / 13), S = 13, device=device)
        self.medium_loss = YOLOLoss(self.anchor_boxes[1] / (416 / 26), S = 26, device=device)
        self.large_loss = YOLOLoss(self.anchor_boxes[0] / (416 / 52), S = 52, device=device)
    
    def forward(self, pred, target):
        s_loss = self.small_loss(pred[0], target[0])
        m_loss = self.medium_loss(pred[1], target[1])
        l_loss = self.large_loss(pred[2], target[2])
        
        return s_loss + m_loss + l_loss