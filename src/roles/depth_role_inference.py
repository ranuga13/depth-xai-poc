import numpy as np

from src.roles.scene_roles import polygon_to_mask


def bbox_xywh_to_xyxy(box):
    x, y, w, h = box

    return [
        x,
        y,
        x + w,
        y + h
    ]


def bbox_intersection_area(
    box_a,
    box_b
):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(
        ax1,
        bx1
    )

    y1 = max(
        ay1,
        by1
    )

    x2 = min(
        ax2,
        bx2
    )

    y2 = min(
        ay2,
        by2
    )

    width = max(
        0,
        x2 - x1
    )

    height = max(
        0,
        y2 - y1
    )

    return (
        width
        * height
    )


def bbox_intersection_ratio(
    target_box,
    candidate_box
):
    intersection = (
        bbox_intersection_area(
            target_box,
            candidate_box
        )
    )

    tx1, ty1, tx2, ty2 = (
        target_box
    )

    target_area = max(
        1,
        (tx2 - tx1)
        * (ty2 - ty1)
    )

    return (
        intersection
        / target_area
    )


def get_median_depth(
    mask,
    depth_map
):
    values = depth_map[
        mask
    ]

    if values.size == 0:
        return np.nan

    values = values[
        np.isfinite(
            values
        )
    ]

    if values.size == 0:
        return np.nan

    return float(
        np.median(
            values
        )
    )


def infer_scene_roles_from_depth(
    target_annotation,
    image_annotations,
    depth_map,
    height,
    width,
    min_bbox_overlap_ratio=0.01,
    min_depth_difference=0.05,
    context_scale=1.5
):
    """
    Infer target / foreground occluder / context
    without using KINS oco_id or ico_id.

    IMPORTANT:
    Current Depth Anything V2 output used in this POC:
    larger raw value = closer to the camera.

    Candidate selection uses:
        1. geometric overlap with target box
        2. candidate depth > target depth
        3. overlap + depth difference scoring

    KINS occlusion-order fields are NOT used here.
    """

    # ======================================================
    # Target
    # ======================================================

    target_visible_mask = polygon_to_mask(
        target_annotation[
            "i_segm"
        ],
        height,
        width
    )

    target_depth = get_median_depth(
        target_visible_mask,
        depth_map
    )


    # Using target amodal bbox only as spatial extent.
    # We are NOT using oco_id / ico_id.
    target_box = bbox_xywh_to_xyxy(
        target_annotation[
            "a_bbox"
        ]
    )


    candidates = []


    # ======================================================
    # Candidate objects
    # ======================================================

    for candidate in image_annotations:

        if (
            candidate["id"]
            == target_annotation["id"]
        ):
            continue


        candidate_visible_mask = (
            polygon_to_mask(
                candidate[
                    "i_segm"
                ],
                height,
                width
            )
        )


        if (
            candidate_visible_mask.sum()
            == 0
        ):
            continue


        candidate_box = (
            bbox_xywh_to_xyxy(
                candidate[
                    "i_bbox"
                ]
            )
        )


        overlap_ratio = (
            bbox_intersection_ratio(
                target_box,
                candidate_box
            )
        )


        if (
            overlap_ratio
            < min_bbox_overlap_ratio
        ):
            continue


        candidate_depth = (
            get_median_depth(
                candidate_visible_mask,
                depth_map
            )
        )


        if np.isnan(
            candidate_depth
        ):
            continue


        depth_difference = (
            candidate_depth
            - target_depth
        )


        is_closer = (
            depth_difference
            > min_depth_difference
        )


        # Simple combined score:
        # geometric overlap × positive depth advantage
        if is_closer:

            score = (
                overlap_ratio
                * depth_difference
            )

        else:

            score = 0.0


        candidates.append(
            {
                "annotation":
                    candidate,

                "annotation_id":
                    candidate["id"],

                "overlap_ratio":
                    float(
                        overlap_ratio
                    ),

                "candidate_depth":
                    float(
                        candidate_depth
                    ),

                "depth_difference":
                    float(
                        depth_difference
                    ),

                "is_closer":
                    bool(
                        is_closer
                    ),

                "score":
                    float(
                        score
                    )
            }
        )


    # ======================================================
    # Foreground candidates only
    # ======================================================

    foreground_candidates = [
        candidate
        for candidate in candidates
        if candidate[
            "is_closer"
        ]
    ]


    predicted_occluder = None


    if foreground_candidates:

        best_candidate = max(
            foreground_candidates,
            key=lambda candidate:
            candidate[
                "score"
            ]
        )

        predicted_occluder = (
            best_candidate[
                "annotation"
            ]
        )


    # ======================================================
    # Occluder mask
    # ======================================================

    occluder_mask = np.zeros(
        (height, width),
        dtype=bool
    )


    if (
        predicted_occluder
        is not None
    ):

        occluder_mask = (
            polygon_to_mask(
                predicted_occluder[
                    "i_segm"
                ],
                height,
                width
            )
        )


    # ======================================================
    # Context region
    # ======================================================

    x, y, w, h = (
        target_annotation[
            "a_bbox"
        ]
    )


    cx = x + w / 2
    cy = y + h / 2

    local_w = (
        w
        * context_scale
    )

    local_h = (
        h
        * context_scale
    )


    x1 = max(
        0,
        int(
            cx
            - local_w / 2
        )
    )

    y1 = max(
        0,
        int(
            cy
            - local_h / 2
        )
    )

    x2 = min(
        width,
        int(
            cx
            + local_w / 2
        )
    )

    y2 = min(
        height,
        int(
            cy
            + local_h / 2
        )
    )


    local_mask = np.zeros(
        (height, width),
        dtype=bool
    )

    local_mask[
        y1:y2,
        x1:x2
    ] = True


    context_mask = (
        local_mask
        & ~target_visible_mask
        & ~occluder_mask
    )


    # ======================================================
    # Return
    # ======================================================

    return {
        "target_mask":
            target_visible_mask,

        "occluder_mask":
            occluder_mask,

        "context_mask":
            context_mask,

        "predicted_occluder":
            predicted_occluder,

        "target_depth":
            target_depth,

        "candidates":
            candidates
    }