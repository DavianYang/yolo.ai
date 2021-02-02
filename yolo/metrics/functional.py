from collections import Counter

import torch

def iou(boxes_preds, boxes_labels, box_format="midpoint"):
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
        
    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_x2 = boxes_preds[..., 3:4]

        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_x2 = boxes_labels[..., 3:4]
    
    x1, y1 = torch.max(box1_x1, box2_x1), torch.max(box1_y1, box2_y1)
    x2, y2 = torch.min(box1_x2, box2_x2), torch.min(box1_y2, box2_y2)
    
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    
    box1_area = abs((box1_x2 - box1_x1) * (box1_y1 - box1_y2))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y1 - box2_y2))
    
    union = box1_area + box2_area - intersection

    return intersection / (union + 1e-6)


def mAP(pred_boxes, true_boxes, iou_threshold=0.5, box_format='corners', num_classes=20):
    avg_precisions = []
    epsilon = 1e-6
    
    # pred_boxes: [[train_idx, class_pred, prob_socre, x1, y1, x2, y2], ...]
    
    for c in range(num_classes):
        detections = [detection for detection in pred_boxes if detection[1] == c]
        ground_truths = [truth_box for truth_box in true_boxes if truth_box[1] == c]
        
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        
        # amount_bboxes = {0: torch.tensor([0, 0, 0]), 1: ...}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
        
        # Sort the prob sore in descending order
        detections.sort(key=lambda x: x[2], reverse=True)
        
        # initialize TP and FP
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_boxes = len(ground_truths)
        
        for det_idx, detection in enumerate(detections):
            # take out only ground truth img
            ground_truth_img = [box for box in ground_truths if box[0] == detection[0]]
            
            no_gt = len(ground_truth_img)
            
            best_iou = 0
            
            for gt_idx, gt in enumerate(ground_truth_img):
                iou = IOU(torch.tensor(detection[3:]), torch.tensor(gt[3:]), box_format=box_format)
                # find best iou
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[det_idx] = 1
                else:
                    FP[det_idx] = 1
                    
            else:
                FP[det_idx] = 1
                
        TP_sum = torch.cumsum(TP, dim=0)
        FP_sum = torch.cumsum(FP, dim=0)
        
        recalls = TP_sum / (total_true_boxes + epsilon)
        precisions = TP_sum / (TP_sum + FP_sum + epsilon) 
        
        recalls = torch.cat((torch.tensor([0]), recalls))
        precisions = torch.cat((torch.tensor([1]), precisions))
        
        avg_precisions.append(torch.trapz(precisions, recalls))
        
    return sum(avg_precisions) / len(avg_precisions)