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