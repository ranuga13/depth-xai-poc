from src.detector.target_matcher import calculate_iou, xywh_to_xyxy


def match_target_to_segment_instance(
    target_annotation,
    target_class,
    instances,
    min_iou=0.30
):
    """
    Match a KINS target to a predicted YOLO segmentation instance.

    KINS is used only to identify the evaluation target.
    The returned geometry comes from YOLO11n-seg.
    """

    target_box = xywh_to_xyxy(
        target_annotation["i_bbox"]
    )

    best_instance = None
    best_iou = 0.0

    for instance in instances:

        if instance["class_name"] != target_class:
            continue

        if instance["mask"] is None:
            continue

        iou = calculate_iou(
            target_box,
            instance["bbox"]
        )

        if iou > best_iou:
            best_iou = iou
            best_instance = instance

    if best_instance is None:
        return None, 0.0

    if best_iou < min_iou:
        return None, best_iou

    return best_instance, best_iou