import xml.etree.ElementTree as ET
from typing import Callable, List, Optional, Tuple

import os
import copy
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
from torchvision.ops.boxes import box_iou

from yolo.utils.utils import IOU, corner_to_center
# ImageFile.LOAD_TRUNCATED_IMAGES = True

VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

class VOCDataset(VOCDetection):
    def __init__(
        self,
        anchor_boxes: list,
        root: str = './datasets',
        year: str = '2012',
        image_set: str = 'train',
        download: bool = False,
        transforms: Optional[Callable] = None,
        num_classes: int = 20,
    ):
        super().__init__(root, year, image_set, download)
        self.transforms = transforms
        self.anchor_boxes = anchor_boxes
        self.num_classes = num_classes
        
    def _generate_label_matrix(self, S, anchor_boxes, targets):
        label_matrix = torch.zeros(
            (S, S, len(anchor_boxes), 5 + self.num_classes), dtype=torch.float64
        )
        
        for target in targets:
            class_label, xmin, ymin, xmax, ymax = target
            class_label = int(class_label)
            
            x, y, w, h = corner_to_center(xmin, ymin, xmax, ymax)
            
            i, j = int(S * x), int(S * y)
            x_cell, y_cell = S * x, S * y
            # We need iou of anchor box and bbox as if they have same xmin and ymin, as only the width and height matters while assigning the bbox to anchor box.
            anchor_boxes[:, 0] = xmin
            anchor_boxes[:, 1] = ymin
            anchor_boxes[:, 2] = xmin + anchor_boxes[:, 2] / 2
            anchor_boxes[:, 3] = ymin + anchor_boxes[:, 3] / 2

            width_cell, height_cell = (w * S, h * S)
            
            ious = box_iou(
                anchor_boxes,
                torch.tensor([xmin, ymin, xmax, ymax]).unsequeeze(0).float()
            )
            
            _, max_idx = ious.max(0)
            
            box_coordinate = torch.tensor([x_cell, y_cell, width_cell, height_cell])
            # set box_coordinate
            label_matrix[j, i, max_idx[0], :4] = box_coordinate
            # set confidence score
            label_matrix[j, i, max_idx[0], 4] = 1
            # set one hot coding for class label
            label_matrix[j, i, max_idx[0], 5 + class_label] = 1
        return label_matrix
            
             
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        
        root_ = ET.parse(self.annotations[index]).getroot()
        targets = []
        for obj in root_.iter("object"):
            target = [0]
            target.append(VOC_CLASSES.index(obj.find("name").text))
            bbox = obj.find('bndbox')
            for xyxy in ("xmin", "ymin", "xmax", "ymax"):
                target.append(bbox.find(xyxy).text)
            targets.append(target)
        targets = torch.tensor(targets)
        
        if self.transforms:
            pass
        
        small_label_matrix = self._generate_label_matrix(
            13, copy.deepcopy(self.anchor_boxes)[2] / (416 / 13)
        )
        medium_label_matrix = self._generate_label_matrix(
            26, copy.deepcopy(self.anchor_boxes)[1] / (416 / 26)
        )
        large_label_matrix = self._generate_label_matrix(
            52, copy.deepcopy(self.anchor_boxes)[0] / (416 / 52)
        )
        
        return small_label_matrix, medium_label_matrix, large_label_matrix
        
        

class YOLODataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        anchors,
        image_size = 416,
        S = [13, 26, 52],
        C = 20,
        transform = None
    ) -> None:
        super().__init__()
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5

    def __len__(self) -> int:
        return len(self.annotations)
    
    
    def __getitem__(self, index: int):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndim=2), 4, axis=1).tolist()
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))
        
        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]
        
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        
        for box in bboxes:
            iou_anchors = IOU(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False, False, False]
            
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                
                S = self.S[scale_idx]
                # ycell, xcell
                i, j = int(S*y), int(S*x)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S*x - j, S*y - i
                    width_cell, height_cell = (
                        width * S,
                        height * S
                    )
                    box_coordinate = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinate
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True
                
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1 # ignore this prediction
                    
            return image, tuple(targets)
                
        
        