from collections import namedtuple
from yolo.config.cfg import NUM_CLASSES, NUM_ANCHORS

convblock = namedtuple("Conv", ["kernel_size", "filters", "stride", "padding"])
convblock_without_bn = namedtuple("ConvWithoutBN", ["kernel_size", "filters", "stride", "padding"])
maxpool = namedtuple("MaxPool", ["kernel_size", "stride"])
repeat = namedtuple("Repeat", ["blocks", "nums"])
repeat_resblock = namedtuple("RepeatWithResidual", ["blocks", "nums"])
scale = namedtuple("ScalePred", ["num_classes", "num_anchors"])

darknet_cfg = [
    convblock(7, 64, 2, 3),
    maxpool(2, 2),
    convblock(3, 192, 1, 1),
    maxpool(2, 2),
    convblock(1, 128, 1, 0),
    convblock(3, 256, 1, 1),
    convblock(1, 256, 1, 0),
    convblock(3, 512, 1, 1),
    maxpool(2, 2),
    
    repeat([convblock(1, 256, 1, 0), convblock(3, 512, 1, 1)], 4),
    convblock(1, 512, 1, 0),
    convblock(3, 1024, 1, 1),
    maxpool(2, 2),
    repeat([convblock(1, 512, 1, 0), convblock(3, 1024, 1, 1)], 2),
    convblock(3, 1024, 2, 1),
    convblock(3, 1024, 1, 1),
    convblock(3, 1024, 1, 1),
]

darknet19_cfg_head = [
    convblock(3, 32, 1, 1),
    maxpool(2, 2),
    convblock(3, 64, 1, 1),
    maxpool(2, 2),
    convblock(3, 128, 1, 1),
    convblock(1, 64, 1, 0),
    convblock(3, 128, 1, 1),
    maxpool(2, 2),
    convblock(3, 256, 1, 1),
    convblock(1, 128, 1, 0),
    convblock(3, 256, 1, 1),
    maxpool(2, 2),
    convblock(3, 512, 1, 1),
    convblock(1, 256, 1, 0),
    convblock(3, 512, 1, 1),
    convblock(1, 256, 1, 0),
    convblock(3, 512, 1, 1),
]

darknet19_cfg_tail = [
    maxpool(2, 2),
    convblock(3, 1024, 1, 1),
    convblock(1, 512, 1, 0),
    convblock(3, 1024, 1, 1),
    convblock(1, 512, 1, 0),
    convblock(3, 1024, 1, 1),
    convblock(3, 1024, 1, 1),
    convblock(3, 1024, 1, 1),
]

darknet53_base_cfg = [
    [   
        convblock(3, 32, 1, 1),
        convblock(3, 64, 2, 1),
        repeat_resblock([convblock(1, 32, 1, 0), convblock(3, 64, 1, 1)], 1),
        convblock(3, 128, 2, 1),
        repeat_resblock([convblock(1, 64, 1, 0), convblock(3, 128, 1, 1)], 2)
    ],
    [
        convblock(3, 256, 2, 1),
        repeat_resblock([convblock(1, 128, 1, 0), convblock(3, 256, 1, 1)], 8)
    ],
    [
        convblock(3, 512, 2, 1),
        repeat_resblock([convblock(1, 256, 1, 0), convblock(3, 512, 1, 1)], 8)
    ],
    [
        convblock(3, 1024, 2, 1),
        repeat_resblock([convblock(1, 512, 1, 0), convblock(3, 1024, 1, 1)], 4)
    ]
]

small_scale_cfg = [
    [repeat([convblock(1, 512, 1, 0), convblock(3, 1024, 1, 1)], 2)],
    [convblock(1, 512, 1, 0)],
    [scale(NUM_CLASSES, NUM_ANCHORS)],
]

medium_scale_cfg = [
    [repeat([convblock(1, 256, 1, 0), convblock(3, 512, 1, 1)], 2)],
    [convblock(1, 256, 1, 0)],
    [scale(NUM_CLASSES, NUM_ANCHORS)]
]

large_scale_cfg = [
    repeat([convblock(1, 128, 1, 0), convblock(3, 256, 1, 1)], 2),
    convblock(1, 128, 1, 0),
    scale(NUM_CLASSES, NUM_ANCHORS)
]