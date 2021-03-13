import torch


class AnchorGenerator(object):
    def __init__(
        self,
        size_base,
        scales=[8],
        ratios=[0.5, 1, 2]
    ):
        self.size_base = size_base
        self.scales = torch.Tensor(scales)
        self.ratios = torch.Tensor(ratios)
        self.base_anchors = self._generateBaseAnchors()
        
        
    def _generateBaseAnchors(self):
        w, h = self.size_base[1], self.size_base[0]
        
        
        
    