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
    image_annotations,
    height,
    width,
    min_overlap_pixels=10
):
    target_oco_id = target_annotation["oco_id"]
    target_ico_id = target_annotation["ico_id"]

    # Full target shape
    target_amodal_mask = polygon_to_mask(
        target_annotation["a_segm"],
        height,
        width
    )

    # Visible target shape
    target_visible_mask = polygon_to_mask(
        target_annotation["i_segm"],
        height,
        width
    )

    # The part of the target that is hidden
    target_hidden_mask = (
        target_amodal_mask
        & ~target_visible_mask
    )

    candidates = []

    for ann in image_annotations:

        if ann["id"] == target_annotation["id"]:
            continue

        # Must belong to the same occlusion group
        if ann["oco_id"] != target_oco_id:
            continue

        # Must be in front of the target
        if ann["ico_id"] >= target_ico_id:
            continue

        candidate_visible_mask = polygon_to_mask(
            ann["i_segm"],
            height,
            width
        )

        # How much of this candidate lies over
        # the hidden part of the target?
        overlap_pixels = (
            candidate_visible_mask
            & target_hidden_mask
        ).sum()

        if overlap_pixels < min_overlap_pixels:
            continue

        candidates.append(
            {
                "annotation": ann,
                "overlap_pixels": int(
                    overlap_pixels
                )
            }
        )

    if not candidates:
        return None

    # Prefer the candidate that actually covers
    # the largest amount of hidden target area.
    best_candidate = max(
        candidates,
        key=lambda item: (
            item["overlap_pixels"],
            item["annotation"]["ico_id"]
        )
    )

    return best_candidate["annotation"]


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
        target_annotation=target_annotation,
        image_annotations=image_annotations,
        height=height,
        width=width
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