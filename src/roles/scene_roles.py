import numpy as np
from pycocotools import mask as mask_utils


def polygon_to_mask(segmentation, height, width):
    rles = mask_utils.frPyObjects(
        segmentation,
        height,
        width
    )

    rle = mask_utils.merge(rles)

    mask = mask_utils.decode(rle)

    return mask.astype(bool)


def find_foreground_occluder(
    target_annotation,
    image_annotations
):
    target_oco_id = target_annotation["oco_id"]
    target_ico_id = target_annotation["ico_id"]

    candidates = [
        ann
        for ann in image_annotations
        if ann["oco_id"] == target_oco_id
        and ann["ico_id"] < target_ico_id
    ]

    if not candidates:
        return None

    # Choose the closest object in front
    occluder = max(
        candidates,
        key=lambda ann: ann["ico_id"]
    )

    return occluder


def create_context_mask(
    target_annotation,
    target_mask,
    occluder_mask,
    height,
    width,
    scale=1.5
):
    x, y, w, h = target_annotation["a_bbox"]

    cx = x + w / 2
    cy = y + h / 2

    new_w = w * scale
    new_h = h * scale

    x1 = max(
        0,
        int(cx - new_w / 2)
    )

    y1 = max(
        0,
        int(cy - new_h / 2)
    )

    x2 = min(
        width,
        int(cx + new_w / 2)
    )

    y2 = min(
        height,
        int(cy + new_h / 2)
    )

    context_mask = np.zeros(
        (height, width),
        dtype=bool
    )

    context_mask[y1:y2, x1:x2] = True

    context_mask[target_mask] = False

    if occluder_mask is not None:
        context_mask[occluder_mask] = False

    return context_mask


def build_scene_roles(
    target_annotation,
    image_annotations,
    height,
    width,
    context_scale=1.5
):
    target_mask = polygon_to_mask(
        target_annotation["i_segm"],
        height,
        width
    )

    occluder_annotation = find_foreground_occluder(
        target_annotation,
        image_annotations
    )

    if occluder_annotation is not None:
        occluder_mask = polygon_to_mask(
            occluder_annotation["i_segm"],
            height,
            width
        )
    else:
        occluder_mask = np.zeros(
            (height, width),
            dtype=bool
        )

    context_mask = create_context_mask(
        target_annotation,
        target_mask,
        occluder_mask,
        height,
        width,
        scale=context_scale
    )

    return {
        "target_mask": target_mask,
        "occluder_mask": occluder_mask,
        "context_mask": context_mask,
        "occluder_annotation": occluder_annotation
    }