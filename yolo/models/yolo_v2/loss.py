import torch
from torch import nn
from torch import Tensor
from torch._C import device
from torch.nn import functional as F

from yolo.utils.utils import IOU

class YOLOv2Loss(nn.Module):
    def __init__(self, S: int, B: int, C: int, device: str, anchor: Tensor) -> None:
        super().__init__()
        self.S, self.B, self.C = S, B, C
        self.mse = nn.MSELoss(reduction="sum")
        self.anchor = anchor
        
    def forward(self, pred, target):
        mask = target[..., 4:5]
        boxes = mask * pred
        cell_idx = torch.arange(13, device=device)
        
        bx = mask * torch.sigmoid(
            pred[..., 0:1]
        ) + mask * cell_idx.view([1, 1, -1, 1, 1])
        by = mask * torch.sigmoid(
            pred[..., 1:2]
        ) + mask * cell_idx.view([1, -1, 1, 1, 1])
        bw = (
            mask
            * self.anchor[:, 2].view([1, 1, 1, -1, 1])
            * mask
            * torch.exp(pred[..., 2:3])
        )
        bh = (
            mask
            * self.anchor[:, 3].view([1, 1, 1, -1, 1])
            * mask
            * torch.exp(pred[..., 3:4])
        )
        
        ious = IOU(
            torch.cat([bx, by, bw, bh], dim=1),
            target[..., :2]
        )
        
        xy_loss = self.mse(torch.cat([bx, by], dim=-1), target[..., :2])
        bwbh = torch.cat([bw, bh], dim=-1)
        wh_loss = self.mse(
            torch.sqrt(torch.abs(bwbh) + 1e-32),
            torch.sqrt(torch.abs(target[..., 2:4]) + 1e-32)
        )
        
        obj_loss = self.mse(
            mask,
            mask * ious * torch.sigmoid(boxes[..., 4:5])
        )
        no_obj_loss = self.mse(
            (1 - mask),
            (
                ((1 - mask) * (1 - torch.sigmoid(pred[..., 4:5])))
                * ((ious.max[-1][0] < 0.6).int().unsqueeze(-1))
            )
        )
        
        class_loss = F.nll_loss()