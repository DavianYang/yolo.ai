import xml.etree.ElementTree as ET
from typing import Callable, Optional

import copy
import numpy as np
from PIL import Image

import torch
from torchvision.datasets import VOCDetection
from torchvision.ops.boxes import box_iou

class VOCDataset(VOCDetection):
    def __init__(
        self,
        classes: list,
        anchor_boxes: list,
        root: str = './datasets',
        year: str = '2012',
        image_set: str = 'train',
        download: bool = False,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, year, image_set, download)
        self.classes = classes
        self.num_classes = len(classes)
        self.transforms = transforms
        self.anchor_boxes = torch.tensor(anchor_boxes)
     
    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        
        root_ = ET.parse(self.annotations[index]).getroot()
        targets = []
        for obj in root_.iter("object"):
            target = []
            target.append(self.classes.index(obj.find("name").text))
            bbox = obj.find('bndbox')
            for xyxy in ("xmin", "ymin", "xmax", "ymax"):
                target.append(int(bbox.find(xyxy).text))
            targets.append(target)
        targets = torch.tensor(targets)
        image = np.array(image)
        if self.transforms:
            output_labels_list = targets[:, 0].int().tolist()
            if type(output_labels_list) == str:
                output_labels_list = [output_labels_list]
            transformed_items = self.transforms(
                image=image, bboxes=targets[:, 1:], class_labels=output_labels_list
            )
            image = transformed_items["image"]
            boxes = transformed_items["bboxes"]
            class_labels = transformed_items["class_labels"]
        
        small_label_matrix = self._generate_label_matrix(
            13, boxes, class_labels, copy.deepcopy(self.anchor_boxes)[2] / (416 / 13)
        )
        medium_label_matrix = self._generate_label_matrix(
            26, boxes, class_labels, copy.deepcopy(self.anchor_boxes)[1] / (416 / 26)
        )
        large_label_matrix = self._generate_label_matrix(
            52, boxes, class_labels, copy.deepcopy(self.anchor_boxes)[0] / (416 / 52)
        )
        return image, (small_label_matrix, medium_label_matrix, large_label_matrix)
    
    
    def _generate_label_matrix(self, S, boxes, class_labels, anchor_boxes):
        label_matrix = torch.zeros(
            (S, S, len(anchor_boxes), 5 + self.num_classes), dtype=torch.float64
        )

        for box, class_label in zip(boxes, class_labels):
            xmin, ymin, xmax, ymax = box
            
            x = xmin / 416
            y = ymin / 416
            w = (xmax - xmin) / 416
            h = (ymax - ymin) / 416
            
            i, j = int(S * x), int(S * y)
            
            x_cell, y_cell = S * x - i, S * y - j
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