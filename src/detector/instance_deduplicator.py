import numpy as np

from src.detector.target_matcher import calculate_iou


def calculate_mask_iou(mask_a, mask_b):
    if mask_a is None or mask_b is None:
        return 0.0

    intersection = np.logical_and(
        mask_a,
        mask_b
    ).sum()

    union = np.logical_or(
        mask_a,
        mask_b
    ).sum()

    if union == 0:
        return 0.0

    return float(
        intersection / union
    )


def deduplicate_instances(
    instances,
    bbox_iou_threshold=0.80,
    mask_iou_threshold=0.70
):
    """
    Remove likely duplicate instance-segmentation predictions.

    A candidate is treated as a duplicate when BOTH:

        bbox IoU >= bbox_iou_threshold
        mask IoU >= mask_iou_threshold

    The higher-confidence prediction is retained.

    This is intentionally class-agnostic because YOLO may
    describe the same physical vehicle as car/truck/etc.
    """

    if not instances:
        return [], []

    # Highest-confidence predictions first
    sorted_instances = sorted(
        instances,
        key=lambda item: item["confidence"],
        reverse=True
    )

    kept = []
    removed = []

    for candidate in sorted_instances:

        duplicate_of = None
        duplicate_bbox_iou = 0.0
        duplicate_mask_iou = 0.0

        for existing in kept:

            bbox_iou = calculate_iou(
                candidate["bbox"],
                existing["bbox"]
            )

            mask_iou = calculate_mask_iou(
                candidate["mask"],
                existing["mask"]
            )

            if (
                bbox_iou >= bbox_iou_threshold
                and
                mask_iou >= mask_iou_threshold
            ):
                duplicate_of = existing
                duplicate_bbox_iou = bbox_iou
                duplicate_mask_iou = mask_iou
                break

        if duplicate_of is None:

            kept.append(
                candidate
            )

        else:

            removed.append(
                {
                    "instance": candidate,
                    "duplicate_of": duplicate_of,
                    "bbox_iou": float(
                        duplicate_bbox_iou
                    ),
                    "mask_iou": float(
                        duplicate_mask_iou
                    )
                }
            )

    return kept, removed