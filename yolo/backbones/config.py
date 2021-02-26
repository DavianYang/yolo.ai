from collections import namedtuple

conv_config = namedtuple("ConvConfig", ["kernel_size", "filters", "stride", "padding"])
max_config = namedtuple("MaxPoolConfig", ["kernel_size", "stride"])
repeat = namedtuple("Repeat", ["blocks", "n"])

darknet_cfg = [
    conv_config(7, 64, 2, 3),
    max_config(2, 2),
    conv_config(3, 192, 1, 1),
    max_config(2, 2),
    conv_config(1, 128, 1, 0),
    conv_config(3, 256, 1, 1),
    conv_config(1, 256, 1, 0),
    conv_config(3, 512, 1, 1),
    max_config(2, 2),
    
    repeat([conv_config(1, 256, 1, 0), conv_config(3, 512, 1, 1)], 4),
    conv_config(1, 512, 1, 0),
    conv_config(3, 1024, 1, 1),
    max_config(2, 2),
    repeat([conv_config(1, 512, 1, 0), conv_config(3, 1024, 1, 1)], 2),
    conv_config(3, 1024, 2, 1),
    conv_config(3, 1024, 1, 1),
    conv_config(3, 1024, 1, 1),
]

darknet19_cfg_head = [
    conv_config(3, 32, 1, 1),
    max_config(2, 2),
    conv_config(3, 64, 1, 1),
    max_config(2, 2),
    conv_config(3, 128, 1, 1),
    conv_config(1, 64, 1, 0),
    conv_config(3, 128, 1, 1),
    max_config(2, 2),
    conv_config(3, 256, 1, 1),
    conv_config(1, 128, 1, 0),
    conv_config(3, 256, 1, 1),
    max_config(2, 2),
    conv_config(3, 512, 1, 1),
    conv_config(1, 256, 1, 0),
    conv_config(3, 512, 1, 1),
    conv_config(1, 256, 1, 0),
    conv_config(3, 512, 1, 1),
]

darknet19_cfg_tail = [
    max_config(2, 2),
    conv_config(3, 1024, 1, 1),
    conv_config(1, 512, 1, 0),
    conv_config(3, 1024, 1, 1),
    conv_config(1, 512, 1, 0),
    conv_config(3, 1024, 1, 1),
    conv_config(3, 1024, 1, 1),
    conv_config(3, 1024, 1, 1),
]