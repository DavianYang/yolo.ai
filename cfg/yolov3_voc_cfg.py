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