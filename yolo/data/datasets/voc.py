import xml.etree.ElementTree as ET
from typing import Callable, List, Optional, Tuple

from PIL import Image

import torch
from torchvision.datasets import VOCDetection

from yolo.models.metrics.functional import iou_width_height

class VOCDataset(VOCDetection):
    def __init__(
        self,
        anchor_boxes: List[Tuple[int]],
        classes: List[str],
        grid_size: List[int] = [52, 26, 13],
        root: str = './datasets',
        year: str = '2012',
        image_set: str = 'train',
        download: bool = False,
        transforms: Optional[Callable] = None
    ) -> None:
        super().__init__(root, year, image_set, download)
        self.anchor_boxes = torch.tensor(anchor_boxes[0] + anchor_boxes[1] + anchor_boxes[2])
        self.classes = classes
        self.grid_size = grid_size
        
        self.num_classes = len(classes)
        self.num_anchors = len(anchor_boxes)
        self.num_anchors = self.anchor_boxes.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        
        self.ignore_iou_thresh = 0.5
        
        self.transforms = transforms
        
                    
    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        targets = self._get_targets(index)
        class_labels = targets[:, 0]
        boxes = targets[:, 1:5]
        
        if type(class_labels) == str:
            class_labels = [class_labels]
        
        if self.transforms:
            image, boxes = self.transforms(image, boxes)
        
        label_matrix = self._generate_label_matrix(image, boxes, class_labels)
        
        return image, tuple(label_matrix)
    
    def _get_targets(self, index):
        targets = []
        root_ = ET.parse(self.annotations[index]).getroot()
        for obj in root_.iter("object"):
            target = []
            target.append(self.classes.index(obj.find("name").text))
            bbox = obj.find('bndbox')
            for xyxy in ("xmin", "ymin", "xmax", "ymax"):
                target.append(int(bbox.find(xyxy).text))
            targets.append(target)
        targets = torch.tensor(targets)
        return targets
    
    def _generate_label_matrix(self, image, boxes, class_labels):
        label_matrix = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.grid_size]
        width, height = image.size
        
        for box, class_label in zip(boxes, class_labels):
            xmin, ymin, xmax, ymax = box
            
            x, y = xmin / width, ymin / height
            w, h = (xmax - xmin) / width, (ymax - ymin) / height
            
            iou_anchors = iou_width_height(torch.tensor([w, h]), self.anchor_boxes)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            has_anchor = [False] * 3
            
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                
                S = self.grid_size[scale_idx]
                i, j = int(S * y), int(S * x)
                
                anchor_taken = label_matrix[scale_idx][anchor_on_scale, i, j, 0]
                
                if not anchor_taken and not has_anchor[scale_idx]:
                    label_matrix[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i
                    
                    width_cell, height_cell = (
                        w * S,
                        h * S,
                    )
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    
                    label_matrix[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    label_matrix[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    
                    has_anchor[scale_idx] = True
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    label_matrix[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction
        
        return label_matrix # gird_size, grid_size, 3, 4 + 1 + classes