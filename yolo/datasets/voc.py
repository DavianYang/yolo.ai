from xml.etree import ElementTree as ET
from typing import Callable, Optional

import copy
from PIL import Image, ImageFile

import torch
from torchvision.datasets import VOCDetection
from torchvision.ops.boxes import box_iou

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
        self.anchor_boxes = torch.tensor(anchor_boxes)
        self.num_classes = num_classes
        
    def _generate_label_matrix(self, S, targets, anchor_boxes):
        label_matrix = torch.zeros(
            (S, S, len(anchor_boxes), 5 + self.num_classes), dtype=torch.float64
        )

        for target in targets:
            class_label, xmin, ymin, xmax, ymax = target
            
            x = xmin
            y = ymin
            w = xmax - xmin
            h = ymax - ymin
            
            x = xmin / 416
            y = ymin / 416
            w = (xmax - xmin) / 416
            h = (ymax - ymin) / 416
            
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
                torch.tensor([xmin, ymin, xmax, ymax]).unsqueeze(0).float()
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
            target = []
            target.append(VOC_CLASSES.index(obj.find("name").text))
            bbox = obj.find('bndbox')
            for xyxy in ("xmin", "ymin", "xmax", "ymax"):
                target.append(int(bbox.find(xyxy).text))
            targets.append(target)
        targets = torch.tensor(targets)
        
        if self.transforms:
            pass
        
        small_label_matrix = self._generate_label_matrix(
            13, targets, copy.deepcopy(self.anchor_boxes)[6:9] / (416 / 13)
        )
        medium_label_matrix = self._generate_label_matrix(
            26, targets, copy.deepcopy(self.anchor_boxes)[3:6] / (416 / 26)
        )
        large_label_matrix = self._generate_label_matrix(
            52, targets, copy.deepcopy(self.anchor_boxes)[:3] / (416 / 52)
        )
        return img, (small_label_matrix, medium_label_matrix, large_label_matrix)