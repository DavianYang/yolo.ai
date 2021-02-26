from typing import Dict, List, Any

import torch
import torch.nn as nn
from torch import Tensor

from yolo.modules.modules import ConvBlock, ResBlock, ReorgBlock
from yolo.backbones.config import darknet_cfg, darknet19_cfg_head, darknet19_cfg_tail

class DarkNet(nn.Module):
    def __init__(
        self,
        cfg: List[tuple],
        in_channels: int = 3
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.features = self._make_layers(cfg)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.features(x)
    
    def _make_layers(self, cfg: List[tuple]) -> nn.Sequential:
        layers = []
        in_channels = self.in_channels
        
        for x in cfg:
            if "Conv" in str(type(x)):
                layers += [ConvBlock(in_channels, x.filters, kernel=x.kernel_size, stride=x.stride, padding=x.padding)]
                
                in_channels = x.filters
            elif "Max" in str(type(x)):
                layers += [nn.MaxPool2d(kernel_size=x.kernel_size, stride=x.stride)]
            elif "Repeat" in str(type(x)):
                convs = x.blocks
                num_repeats = x.nums
                
                for _ in range(num_repeats):
                    for conv in convs:
                        layers += [ConvBlock(in_channels, conv.filters, kernel=conv.kernel_size, stride=conv.stride, padding=conv.padding)]
                        
                        in_channels = conv.filters
        return nn.Sequential(*layers)
            
def _darknet(
    cfg: List[tuple],
    in_channels: int = 3
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