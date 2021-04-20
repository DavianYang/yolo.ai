model = dict(
    type='YOLOv3',
    pretrained=None,
    backbone=dict(
        type='DarkNet53',
        in_channels=3
    ),
    head=dict(
        type='MultiScaleDetector',
        num_classes=20,
        anchors=None,
        nms_type='nms',
        nms_thr=None,
        yolo_loss_type=None,
        ignore_thr=None,
        box_loss=dict(type='IOU_Loss', iou_type=None),
        confidence_loss=dict(type='Conf_Loss'),
        class_loss=dict(type='Class Loss')
    )
)

img_scale = None

training_pipeline = []
testing_pipeline = []
val_pipeline = []

yolov3_anchors = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]

classes = [
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

data = dict()