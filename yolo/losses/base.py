import torch
from torch import nn
from torch.nn import functional as F

from yolo.utils.utils import IOU

class YOLOLoss(nn.Module):
    def __init__(self, anchor_boxes: list, grid_size: int, device: str):
        super().__init__()
        self.anchor_boxes = anchor_boxes
        self.mse = nn.MSELoss()
        self.anchor_boxes = anchor_boxes.to(device)
        self.grid_size = grid_size
        self.device = device
        
    def forward(self, pred, target):
        pred = pred.permute(0, 2, 3, 1, 4)
        exist_mask = target[..., 4:5]
        existing_boxes = exist_mask * pred
        cell_idx = torch.arange(self.grid_size, device=self.device)
        
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