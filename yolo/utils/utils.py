import torch

def corner_to_center(xmin, ymin, xmax, ymax):
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin
    return cx, cy, w, h

def center_to_corner(cx, cy, w, h):
    xmin, ymin = cx - 0.5 * w, cy - 0.5 * h
    xmax, ymax = cx + 0.5 * w, cy + 0.5 * h
    return xmin, ymin, xmax, ymax

def cells_to_bboxes(preds, anchors, S, is_pred):
    batch_size = preds.shape[0]
    num_anchors = len(anchors)
    
    x_pred, y_pred = preds[..., 1:2], preds[..., 2:3]
    w_pred, h_pred = preds[..., 3:4], preds[..., 4:5]
    
    if is_pred:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        x_pred, y_pred = torch.sigmoid(x_pred), torch.sigmoid(y_pred)
        w_pred, h_pred = torch.exp(w_pred) * anchors, torch.exp(h_pred) * anchors
    
    scores = preds[..., 0:1]
    best_class = preds[..., 5:6]
    
    cell_indices = (
        torch.arange(S)
        .repeat(batch_size, num_anchors, S, 1)
        .unsqueeze(-1)
        .to(preds.device)
    )
    
    x = 1 / S * (x_pred + cell_indices)
    y = 1 / S * (y_pred + cell_indices.permute(0, 1, 3, 2, 4))
    w, h = 1 / S * w_pred, 1 / S * h_pred
    
    converted_bboxes = torch.cat((best_class, scores, x, y, w, h), dim=-1).reshape(batch_size, num_anchors * S * S, 6)
    return converted_bboxes.tolist()
    