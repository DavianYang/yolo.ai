from inspect import isfunction
from torch import nn

# https://github.com/osmr/imgclsmob/blob/68335927ba27f2356093b985bada0bc3989836b1/pytorch/pytorchcv/models/common.py
def get_activation_layer(activation):
    assert (activation is not None)
    if isfunction(activation):
        return activation()
    elif isinstance(activation, str):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "leakyrelu":
            return nn.LeakyReLU(0.1, inplace=True)
        elif activation == "silu":
            return nn.SiLU()
        else:
            raise NotImplementedError()
    else:
        assert (isinstance(activation, nn.Module))
        return activation