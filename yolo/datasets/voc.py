from typing import Optional, Callable, Tuple, List

import torch
import numpy as np

from PIL import Image
import xml.etree.ElementTree as ET
from torchvision.datasets import VOCDetection

class VOCDetection(VOCDetection):
    def __init__(
        self,
        root: str = './datasets',
        year: str = '2012',
        image_set: str = 'train',
        download: bool = False,
        transform: Optional[Callable] = None
        split: int = 7,
        box: int = 2,
        classes: int = 20
    ):
        super().__init__(root, year, image_set, download)
        self.transform = transform
        self.split = split
        self.box = box
        self.classes = classes
        
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        image = Image.open(self.images[index])
        root_ = ET.parse(self.annotations[index]).getroot()
        targets = []
        for obj in root_.iter('object'):
            target = []
            bbox = obj.find('bndbox')
            target.append(CLASSES.index(obj.find('name').text))
            for xyxy in ('xmin', 'ymin', 'xmax', 'ymax'):
                target.append(int(bbox.find(xyxy).text))
        targets.append(target)
        
        targets = np.array(targets, dtype=np.float32)
        
        label_matrix = torch.zeros((self.split, self.split, self.box * 5 + self.classes))
        
        for target in targets:
            label, xmin, ymin, xmax, ymax = target.tolist()
            
            width = xmax - xmin
            height = ymax - ymin
            
            i, j = int(self.split * ymin), int(self.split * xmin)
            x_cell, y_cell =  self.split * xmin - j, self.split * ymin - i
            
            width_cell, height_cell = (width * self.split, height  * self.split)
            
            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1
                
                box_coord = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                
                label_matrix[i, j, 21:25] = box_coord
                
                label_matrix[i, j, target[0]] = 1
            
        if self.transform is not None:
            bboxes = targets[..., 1:]
            image, bboxes = self.transform(image, bboxes)
            targets[..., 1:] = bboxes
    
        return image, label_matrix