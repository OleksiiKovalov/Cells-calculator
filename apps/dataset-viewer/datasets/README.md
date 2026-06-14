# Demo Cells Dataset: YOLO Segmentation + COCO

This is a tiny synthetic demo dataset with:
- 3 PNG images, 256x256 pixels
- 5 cell-like objects per image
- 15 total cell instances
- 1 class: `cell`

## Split

- train: 2 images, 10 cells
- val: 1 image, 5 cells

## YOLO segmentation format

Path:

```text
yolo_seg/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml
```

Each YOLO label line has:

```text
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```

Coordinates are normalized to 0..1.

## COCO instance segmentation format

Path:

```text
coco/
  images/
    train/
    val/
  annotations/
    instances_train.json
    instances_val.json
```

COCO coordinates are in pixels.

Both YOLO and COCO annotations describe the same synthetic images and the same 5 cells per image.
