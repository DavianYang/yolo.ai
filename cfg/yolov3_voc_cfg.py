model = dict(
    type='YOLOv3',
    pretrained=None,
    backbone=dict(
        type='DarkNet53',
        in_channels=3
    ),
    head=dict(
        type='YOLOv3Detector',
        num_classes=20,
        anchors=[
            [(10, 13), (16, 30), (33, 23)],
            [(30, 61), (62, 45), (59, 119)],
            [(116, 90), (156, 198), (373, 326)]
        ],
        nms_type='nms',
        nms_thr=0.5,
        yolo_loss_type='yolov3_loss',
        ignore_thr=0.4
    )
)

img_scale = 416

training_pipeline = [
    dict(type='Resize',img_scale=img_scale, letterbox=False),
    dict(type='RandomAffine',degrees=0, translate=0, scale=.5, shear=0.0),#随机放射变换
    dict(type='RandomHSV'),
    dict(type='RandomFlip'),
    dict(type='Normalize'),
    dict(type='ToTensorV2'),
]

testing_pipeline = [
    dict(type='Resize',img_scale=img_scale, letterbox=False),
    dict(type='Normalize'),
    dict(type='ToTensorV2'),
]

val_pipeline = [
    dict(type='Resize',img_scale=img_scale, letterbox=False),
    dict(type='Normalize'),
    dict(type='ToTensorV2'),
]

data = dict(
    batch_size=64,
    num_workers=4,
    voc_classes = [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]
)

optimizer = dict(type='SGD', lr=1e-2, momentum=0.9, weight_decay= 5e-4)