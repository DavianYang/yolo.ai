""" Transform """
from typing import Optional, Union, Tuple
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.nn import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img, bboxes)

        return img, bboxes


class ToTensor:
    def __call__(
        self, 
        img: Image.Image, 
        bboxes: np.ndarray, 
        normalize: bool =True
    ):
        img = np.array(img) 
        img = torch.from_numpy(np.moveaxis(img / (255.0 if img.dtype == np.uint8 else 1), -1, 0).astype(np.float32))
        bboxes = torch.from_numpy(bboxes)
        if normalize:
            F.normalize(img, (0.4574, 0.4385, 0.4064), (0.2704, 0.2676, 0.2814))
        return img, bboxes
    
    def __repr__(self):
        return self.__class__.__name__ + '()'


class Resize(nn.Module):
    def __init__(
        self, 
        size: int = (448, 448), 
        interpolation: Optional[int] = Image.BILINEAR,
    ):
        super().__init__()
        
        self.size = size
        self.interpolation = interpolation
        
    def forward(
        self, 
        img: Union[torch.Tensor, np.ndarray], 
        bboxes: Union[torch.Tensor, np.ndarray]
        ) -> Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
        width, height = self.size
        old_width, old_height = img.size
        
        scale_x = width / old_width
        scale_y = height / old_height
        
        img = F.resize(img, (height, width))
        if isinstance(bboxes, torch.Tensor):
            bboxes[..., 0] = torch.round(scale_x * bboxes[..., 0])
            bboxes[..., 1] = torch.round(scale_y * bboxes[..., 1])
            bboxes[..., 2] = torch.round(scale_x * bboxes[..., 2])
            bboxes[..., 3] = torch.round(scale_y * bboxes[..., 3])
        elif isinstance(bboxes, np.ndarray):
            bboxes[..., 0] = np.rint(scale_x * bboxes[..., 0])
            bboxes[..., 1] = np.rint(scale_y * bboxes[..., 1])
            bboxes[..., 2] = np.rint(scale_x * bboxes[..., 2])
            bboxes[..., 3] = np.rint(scale_y * bboxes[..., 3])
        
        return img, bboxes
    

class RandomHorizontalFlip(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        
    def forward(
        self, 
        img: Union[torch.Tensor, np.ndarray], bboxes: Union[torch.Tensor, np.ndarray]
        ) -> Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
        width, _ = img.size
        
        if torch.rand(1) < self.p:
            img = F.hflip(img)
            bboxes[..., 0] = width - bboxes[..., 0] - 1
            bboxes[..., 2] = width - bboxes[..., 2] - 1
        
        return img, bboxes
    

class RandomVerticalFlip(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        
    def forward(
        self, 
        img: Union[torch.Tensor, np.ndarray], bboxes: Union[torch.Tensor, np.ndarray]
        ) -> Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
        _, height = img.size
        
        if torch.rand(1) < self.p:
            img = F.vflip(img)
            bboxes[..., 1] = height - bboxes[..., 1] + 1
            bboxes[..., 3] = height - bboxes[..., 3] + 1
        
        return img, bboxes