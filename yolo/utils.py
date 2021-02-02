def corner_to_center(boxes):
    x1, y1, x2, y2 = boxes[..., 1:]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes[..., 1:] = cx, cy, w, h
    return boxes


def center_to_corner(boxes):
    cx, cy, w, h = boxes[..., 1:]
    x1, y1 = cx - 0.5 * w, cy - 0.5 * h
    x2, y2 = cx + 0.5 * w, cy + 0.5 * h
    boxes[..., 1:] = x1, y1, x2, y2
    return boxes