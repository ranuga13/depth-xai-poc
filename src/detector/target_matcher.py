def xywh_to_xyxy(box):
    x, y, w, h = box

    return [
        x,
        y,
        x + w,
        y + h
    ]


def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    intersection = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def match_target(
    target_annotation,
    target_class,
    detections,
    min_iou=0.30
):
    kins_box = xywh_to_xyxy(
        target_annotation["i_bbox"]
    )

    best_detection = None
    best_iou = 0.0

    for detection in detections:

        if detection["class_name"] != target_class:
            continue

        iou = calculate_iou(
            kins_box,
            detection["bbox"]
        )

        if iou > best_iou:
            best_iou = iou
            best_detection = detection

    if best_detection is None:
        return None, 0.0

    if best_iou < min_iou:
        return None, best_iou

    return best_detection, best_iou