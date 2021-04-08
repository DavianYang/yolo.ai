from PIL import ImageDraw

def visualize_bbox(image, boxes, labels=None, mode="midpoint"):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    for box in boxes:
        if mode == "midpoints":
            x, y, w, h = box
            xmin, ymin = x * width, y * height
            xmax, ymax = (w * width) + xmin, (h * height) + ymin
        elif mode == "corners":
            xmin, ymin, xmax, ymax = box
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=(50, 168, 82), width=3)
        # draw.text((xmin, ymin - 10), label, (50, 168, 82)) # To implment label text
    return image