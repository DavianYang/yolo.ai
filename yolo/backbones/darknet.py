from typing import List, Any, Tuple

import torch.nn as nn
from torch import Tensor

from yolo.config.darknet_cfg import darknet_cfg, darknet19_cfg_head, darknet19_cfg_tail, darknet53_base_cfg
from yolo.modules.modules import make_layers

class DarkNet(nn.Module):
    def __init__(
        self,
        cfg: List[tuple],
        in_channels: int = 3
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.features = make_layers(cfg)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.features(x)
  
class DarkNet53(nn.Module):
    def __init__(self, cfg: List[tuple]) -> None:
        super().__init__()
        self.part1 = make_layers(cfg[0], 3)
        self.part2 = make_layers(cfg[1], 128)
        self.part3 = make_layers(cfg[2], 256)
        self.part4 = make_layers(cfg[3], 512)
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.part1(x)
        large = self.part2(x)
        medium = self.part3(large)
        small = self.part4(medium)
        return small, medium, large

def _darknet(
    cfg: List[tuple],
    in_channels: int = 3,
    **kwargs: Any
) -> DarkNet:
    model = DarkNet(cfg, in_channels)
    
    return model

def darknet() -> DarkNet:
    return _darknet(cfg=darknet_cfg)
    
def darknet19(mode: str = "full") -> DarkNet:
    if mode == "full":
        cfg = darknet19_cfg_head + darknet19_cfg_tail 
    elif mode == "head":
        cfg = darknet19_cfg_head
    elif mode == "tail":
        cfg = darknet19_cfg_tail
    return _darknet(cfg)

def darknet53() -> DarkNet53:
    return DarkNet53(cfg=darknet53_base_cfg)