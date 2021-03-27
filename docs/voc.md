### Pascal VOC

PASCAL (Pattern Analysis, Statistical Modelling, and Computational Learning) is a Network of Excellence by the EU. They ran the Visual Object Challenge (VOC) from 2005 onwards till 2012.

The file structure obtained after annotations from VoTT is as below.

<div align="center">
    <img src="../images/datasets/pascal_voc_file_structure.png">
</div>

The four components are Annotations, ImageSets, JPEGImages, and pascal_label_map.pbxt.

We will not prepare the dataset from scratch since torchvision provide us with ```VOCDetection``` class. So, I will override the class and adjust for the output

Our Dataset class will be composed of two main function ```__getitem__``` and ```_generate_label_matrix``` to output three scale of label matrix.

```python
# voc.py, class VOCDataset

class VOCDataset(VOCDetection):
    def __init__(
        self,
        classes: list,
        anchor_boxes: list,
        root: str = './datasets',
        year: str = '2012',
        image_set: str = 'train',
        download: bool = False,
        transforms: Optional[Callable] = None
    ) -> None:
        super().__init__(root, year, image_set, download)
        self.classes = classes
        self.num_classes = len(classes)
        self.transforms = transforms
        self.anchor_boxes = torch.tensor(anchor_boxes)
        
    def _generate_label_matrix(self, S, boxes, class_labels, anchor_boxes):
        # ... line 47
        return label_matrix
            
             
    def __getitem__(self, index):
        # ... line 33
        return image, (small_label_matrix, medium_label_matrix, large_label_matrix)
```
--text--
```python
# voc.py, class VOCDataset

def __getitem__(self, index):
    image = Image.open(self.images[index]).convert("RGB")
    
    root_ = ET.parse(self.annotations[index]).getroot()
    targets = []
    for obj in root_.iter("object"):
        target = []
        target.append(VOC_CLASSES.index(obj.find("name").text))
        bbox = obj.find('bndbox')
        for xyxy in ("xmin", "ymin", "xmax", "ymax"):
            target.append(int(bbox.find(xyxy).text))
        targets.append(target)
    targets = torch.tensor(targets)
    image = np.array(image)
```
--text--
```python
# voc.py, class VOCDataset

if self.transforms:
    output_labels_list = targets[:, 0].int().tolist()
    if type(output_labels_list) == str:
        output_labels_list = [output_labels_list]
    transformed_items = self.transforms(
        image=image, bboxes=targets[:, 1:], class_labels=output_labels_list
    )
    image = transformed_items["image"]
    boxes = transformed_items["bboxes"]
    class_labels = transformed_items["class_labels"]
```
For three scale, we will generate 13, 26 and 52 scale of label matrix
```python
# voc.py, class VOCDataset

small_label_matrix = self._generate_label_matrix(
    13, boxes, class_labels, copy.deepcopy(self.anchor_boxes)[2] / (416 / 13)
)
medium_label_matrix = self._generate_label_matrix(
    26, boxes, class_labels, copy.deepcopy(self.anchor_boxes)[1] / (416 / 26)
)
large_label_matrix = self._generate_label_matrix(
    52, boxes, class_labels, copy.deepcopy(self.anchor_boxes)[0] / (416 / 52)
)
```
--text--
```python
# voc.py, class VOCDataset

from torchvision.ops.boxes import box_iou

def _generate_label_matrix(self, S, boxes, class_labels, anchor_boxes):
    for box, class_label in zip(boxes, class_labels):
        xmin, ymin, xmax, ymax = box
        
        x = xmin / 416
        y = ymin / 416
        w = (xmax - xmin) / 416
        h = (ymax - ymin) / 416
        
        # Assign gird cell of i, j
        i, j = int(S * x), int(S * y)
        
        x_cell, y_cell = S * x - i, S * y - j

        anchor_boxes[:, 0] = xmin
        anchor_boxes[:, 1] = ymin
        anchor_boxes[:, 2] = xmin + anchor_boxes[:, 2] / 2
        anchor_boxes[:, 3] = ymin + anchor_boxes[:, 3] / 2

        width_cell, height_cell = (w * S, h * S)
        
        ious = box_iou(
            anchor_boxes,
            torch.tensor([xmin, ymin, xmax, ymax]).unsqueeze(0).float()
        )
        
        _, max_idx = ious.max(0)
        
        box_coordinate = torch.tensor([x_cell, y_cell, width_cell, height_cell])
```
--text--
```python
# voc.py, class VOCDataset

def _generate_label_matrix(self, S, boxes, class_labels, anchor_boxes):
    label_matrix = torch.zeros(
        (S, S, len(anchor_boxes), 5 + self.num_classes), dtype=torch.float64
    )

    # ... line 26

        # set box_coordinate
        label_matrix[j, i, max_idx[0], :4] = box_coordinate
        # set confidence score
        label_matrix[j, i, max_idx[0], 4] = 1
        # set one hot coding for class label
        label_matrix[j, i, max_idx[0], 5 + class_label] = 1
    
    return label_matrix
```

> Check for detail [code](https://github.com/DavianYang/yolo.ai/blob/main/yolo/datasets/voc.py)