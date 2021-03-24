from collections import namedtuple

convblock = namedtuple("Conv", ["kernel_size", "filters", "stride", "padding"])
maxpool = namedtuple("MaxPool", ["kernel_size", "stride"])
repeat = namedtuple("Repeat", ["blocks", "nums"])
repeat_resblock = namedtuple("RepeatWithResidual", ["blocks", "nums"])
scale = namedtuple("ScalePred", [])