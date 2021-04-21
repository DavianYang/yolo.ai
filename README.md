# Introduction
## Models:
- [YOLOv1](https://arxiv.org/abs/1506.02640)
- [YOLOv2](https://arxiv.org/abs/1612.08242)
- [YOLOv3](https://arxiv.org/abs/1804.02767)

## Backbone:
- DarkNet
- EfficientNet

## Custom Data Enchancement method:
- Normalize
- Resize
- RandomHorizontalFlip
- RandomVerticalFlip

# Code structure description
```
yolo.ai
├── cfg             # The directory where model config file is located (darknet, efficientnet, etc)
├── tests           # Implmentation test cases
├── yolo            # YOLO implementation code bases
│   ├── backbones   # All backbone network gathering here
│   ├── datasets    # Contain datasets such as pascal-voc
│   ├── detectors   # Assembly of all types of detectors
│   ├── losses      # The gathering place of all loss functions
│   ├── metrics     # Metrics functions for bounding boxes and losses
│   ├── models      # YOLO model assembly place
│   ├── modules     # Individuals modules for network building
│   ├── transforms  # Custom transforms for Dataset
│   └── utils       # Utilites file for visualization and network
```

# Installation

# Quick start

# References
