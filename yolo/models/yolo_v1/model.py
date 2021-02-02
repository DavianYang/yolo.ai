import torch
import torch.nn as nn

architecture_config = [
    # Tuple: (kernel_size, num_filters, stride, padding)
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    # List: 
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        return self.leakyrelu(self.bn(self.conv(x)))
    
    

class Yolov1(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        **kwargs
    ):
        super().__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)
        
    def forward(self, x):
        x = self.darknet(x)
        torch.flatten(x, start_dim=1)
        return self.fcs(torch.flatten(x, start_dim=1))
    
    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        
        for x in architecture:
            if type(x) == tuple:
                layers += [ConvBlock(in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3])]
                
                in_channels = x[1]
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repats = x[2]
                
                for _ in range(num_repats):
                    layers += [ConvBlock(in_channels, conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3])]
                    layers += [ConvBlock(conv1[1], conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3])]
                    
                    in_channels = conv2[1]
        return nn.Sequential(*layers)
    
    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5))
        )


class YoloLoss(nn.Module):
    def __init__(
        self,
        S = 7,
        B = 2,
        C = 20
    ):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S, self.B, self.C = S, B, C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
        
    
    def forward(self, prediction, target):
        pred = prediction.reshape(-1, self.S, self.S, self.C + self.B * 5)
        
        iou_b1 = IOU(pred[..., 21:25], target[..., 21:25])
        iou_b2 = IOU(pred[..., 26:30], target[..., 26:30])
        
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)])
        
        iou_maxes, bestbox = torch.max(ious, dim=0)
        
        exists_box = target[..., 20].unsequeeze(3)
        
        
        # For box coordinates
        box_preds = exists_box * ( 
            (bestbox * prediction[..., 26:30] + (1 - bestbox) * prediction[..., 21:25])
        )
        
        box_targets = exists_box * target[..., 21:25]
        box_preds[..., 2:4] = torch.sign(box_preds[..., 2:4]) * torch.sqrt(torch.abs(box_preds[..., 2:4] + 1e-6))
        
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        
        box_loss = self.mse(torch.flatten(box_preds, end_dim=-2), torch.flatten(box_targets, end_dim=-2))
        
        # For object loss
        pred_box = (
            bestbox * prediction[..., 25:26] + (1 - bestbox) * prediction[..., 20:21]
        )
        
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21])
        )
        
        # For no object loss
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * prediction[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )
        
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * prediction[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 25:26], start_dim=1)
        )
        
        # For class loss
        class_loss = self.mse(
            torch.flatten(exists_box * prediction[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2)
        )
        
        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )
        
        return loss