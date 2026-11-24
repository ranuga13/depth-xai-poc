import numpy as np


def get_median_depth(
    mask,
    depth_map
):
    values = depth_map[mask]

    if values.size == 0:
        return np.nan

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return np.nan

    return float(
        np.median(values)
    )


# ==========================================================
# Bounding-box helpers
# ==========================================================

def bbox_intersection_area(
    box_a,
    box_b
):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)

    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    width = max(
        0.0,
        x2 - x1
    )

    height = max(
        0.0,
        y2 - y1
    )

    return width * height


def bbox_intersection_ratio(
    reference_box,
    candidate_box
):
    intersection = bbox_intersection_area(
        reference_box,
        candidate_box
    )

    rx1, ry1, rx2, ry2 = (
        reference_box
    )

    reference_area = max(
        1.0,
        (rx2 - rx1)
        * (ry2 - ry1)
    )

    return (
        intersection
        / reference_area
    )


def expand_box(
    box,
    width,
    height,
    scale=1.35
):
    x1, y1, x2, y2 = box

    box_width = (
        x2 - x1
    )

    box_height = (
        y2 - y1
    )

    cx = (
        x1 + x2
    ) / 2

    cy = (
        y1 + y2
    ) / 2

    expanded_width = (
        box_width
        * scale
    )

    expanded_height = (
        box_height
        * scale
    )

    expanded_x1 = max(
        0.0,
        cx
        - expanded_width / 2
    )

    expanded_y1 = max(
        0.0,
        cy
        - expanded_height / 2
    )

    expanded_x2 = min(
        float(width),
        cx
        + expanded_width / 2
    )

    expanded_y2 = min(
        float(height),
        cy
        + expanded_height / 2
    )

    return [
        expanded_x1,
        expanded_y1,
        expanded_x2,
        expanded_y2
    ]


# ==========================================================
# Box distance
# ==========================================================

def bbox_gap_distance(
    box_a,
    box_b
):
    """
    Minimum Euclidean gap between two boxes.

    Returns 0 if boxes touch or overlap.
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    if ax2 < bx1:
        dx = (
            bx1 - ax2
        )

    elif bx2 < ax1:
        dx = (
            ax1 - bx2
        )

    else:
        dx = 0.0


    if ay2 < by1:
        dy = (
            by1 - ay2
        )

    elif by2 < ay1:
        dy = (
            ay1 - by2
        )

    else:
        dy = 0.0


    return float(
        np.sqrt(
            dx * dx
            +
            dy * dy
        )
    )


def normalized_bbox_gap(
    target_box,
    candidate_box
):
    """
    Normalize gap distance by target-box diagonal.
    """

    x1, y1, x2, y2 = (
        target_box
    )

    target_width = max(
        1.0,
        x2 - x1
    )

    target_height = max(
        1.0,
        y2 - y1
    )

    target_diagonal = (
        np.sqrt(
            target_width ** 2
            +
            target_height ** 2
        )
    )

    gap = bbox_gap_distance(
        target_box,
        candidate_box
    )

    return float(
        gap
        / target_diagonal
    )


def proximity_from_gap(
    normalized_gap
):
    """
    Convert gap into a bounded proximity score.

    gap = 0
        -> proximity = 1

    increasing gap
        -> proximity approaches 0
    """

    return float(
        1.0
        / (
            1.0
            + normalized_gap
        )
    )


# ==========================================================
# Context
# ==========================================================

def build_context_mask(
    target_box,
    target_mask,
    occluder_mask,
    height,
    width,
    context_scale=1.5
):
    x1, y1, x2, y2 = (
        target_box
    )

    box_width = (
        x2 - x1
    )

    box_height = (
        y2 - y1
    )

    cx = (
        x1 + x2
    ) / 2

    cy = (
        y1 + y2
    ) / 2

    local_width = (
        box_width
        * context_scale
    )

    local_height = (
        box_height
        * context_scale
    )

    local_x1 = max(
        0,
        int(
            cx
            - local_width / 2
        )
    )

    local_y1 = max(
        0,
        int(
            cy
            - local_height / 2
        )
    )

    local_x2 = min(
        width,
        int(
            cx
            + local_width / 2
        )
    )

    local_y2 = min(
        height,
        int(
            cy
            + local_height / 2
        )
    )

    local_mask = np.zeros(
        (height, width),
        dtype=bool
    )

    local_mask[
        local_y1:local_y2,
        local_x1:local_x2
    ] = True

    return (
        local_mask
        & ~target_mask
        & ~occluder_mask
    )


# ==========================================================
# Fully predicted role inference
# ==========================================================

def infer_roles_from_predicted_instances(
    target_instance,
    instances,
    depth_map,
    height,
    width,
    min_bbox_overlap_ratio=0.01,
    max_normalized_gap=0.36,
    min_depth_difference=0.05,
    context_scale=1.5,
    candidate_region_scale=1.35
):
    """
    Fully predicted target / occluder / context inference.

    Uses:
        YOLO11n-seg masks
        YOLO11n-seg boxes
        YOLO confidence
        Depth Anything depth

    Does NOT use KINS geometry or occlusion order.

    Candidate accepted if either:

        expanded target overlap >= threshold

    OR

        normalized box gap <= threshold

    Candidate must also be closer in predicted depth.

    Final score:

        proximity
        x depth advantage
        x confidence
    """

    # ======================================================
    # Target
    # ======================================================

    target_mask = (
        target_instance[
            "mask"
        ]
    )

    target_box = (
        target_instance[
            "bbox"
        ]
    )

    if target_mask is None:
        raise ValueError(
            "Target instance has no mask."
        )


    target_depth = (
        get_median_depth(
            target_mask,
            depth_map
        )
    )


    if np.isnan(
        target_depth
    ):
        raise ValueError(
            "Target depth is invalid."
        )


    expanded_target_box = (
        expand_box(
            target_box,
            width=width,
            height=height,
            scale=candidate_region_scale
        )
    )


    # ======================================================
    # Candidates
    # ======================================================

    candidates = []


    for candidate in instances:

        if (
            candidate[
                "instance_id"
            ]
            ==
            target_instance[
                "instance_id"
            ]
        ):
            continue


        candidate_mask = (
            candidate[
                "mask"
            ]
        )


        if candidate_mask is None:
            continue


        candidate_box = (
            candidate[
                "bbox"
            ]
        )


        # --------------------------------------------------
        # Expanded-region overlap
        # --------------------------------------------------

        expanded_overlap = (
            bbox_intersection_ratio(
                expanded_target_box,
                candidate_box
            )
        )


        # --------------------------------------------------
        # Spatial gap
        # --------------------------------------------------

        normalized_gap = (
            normalized_bbox_gap(
                target_box,
                candidate_box
            )
        )


        gap_proximity = (
            proximity_from_gap(
                normalized_gap
            )
        )


        # --------------------------------------------------
        # Candidate geometry acceptance
        # --------------------------------------------------

        overlap_valid = (
            expanded_overlap
            >= min_bbox_overlap_ratio
        )


        gap_valid = (
            normalized_gap
            <= max_normalized_gap
        )


        geometry_valid = (
            overlap_valid
            or
            gap_valid
        )


        if not geometry_valid:
            continue


        # --------------------------------------------------
        # Depth
        # --------------------------------------------------

        candidate_depth = (
            get_median_depth(
                candidate_mask,
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


        # --------------------------------------------------
        # Combined proximity
        # --------------------------------------------------

        overlap_proximity = min(
            1.0,
            expanded_overlap
            / max(
                min_bbox_overlap_ratio,
                1e-6
            )
        )


        proximity_score = max(
            overlap_proximity,
            gap_proximity
        )


        confidence = float(
            candidate[
                "confidence"
            ]
        )


        # --------------------------------------------------
        # Final score
        # --------------------------------------------------

        if is_closer:

            score = (
                proximity_score
                * depth_difference
                * confidence
            )

        else:

            score = 0.0


        candidates.append(
            {
                "instance":
                    candidate,

                "instance_id":
                    candidate[
                        "instance_id"
                    ],

                "class_name":
                    candidate[
                        "class_name"
                    ],

                "confidence":
                    confidence,

                "bbox":
                    candidate_box,

                "overlap_ratio":
                    float(
                        expanded_overlap
                    ),

                "normalized_gap":
                    float(
                        normalized_gap
                    ),

                "gap_proximity":
                    float(
                        gap_proximity
                    ),

                "proximity_score":
                    float(
                        proximity_score
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

                "geometry_valid":
                    bool(
                        geometry_valid
                    ),

                "score":
                    float(
                        score
                    )
            }
        )


    # ======================================================
    # Foreground candidates
    # ======================================================

    foreground_candidates = [
        candidate
        for candidate in candidates
        if candidate[
            "is_closer"
        ]
    ]


    selected_candidate = None
    predicted_occluder = None


    if foreground_candidates:

        selected_candidate = max(
            foreground_candidates,
            key=lambda candidate:
                candidate[
                    "score"
                ]
        )


        predicted_occluder = (
            selected_candidate[
                "instance"
            ]
        )


    # ======================================================
    # Occluder mask
    # ======================================================

    occluder_mask = np.zeros(
        (height, width),
        dtype=bool
    )


    if predicted_occluder is not None:

        occluder_mask = (
            predicted_occluder[
                "mask"
            ]
        )


    # ======================================================
    # Context
    # ======================================================

    context_mask = (
        build_context_mask(
            target_box=
                target_box,

            target_mask=
                target_mask,

            occluder_mask=
                occluder_mask,

            height=
                height,

            width=
                width,

            context_scale=
                context_scale
        )
    )


    # ======================================================
    # Return
    # ======================================================

    return {
        "target_mask":
            target_mask,

        "occluder_mask":
            occluder_mask,

        "context_mask":
            context_mask,

        "target_box":
            target_box,

        "expanded_target_box":
            expanded_target_box,

        "target_depth":
            float(
                target_depth
            ),

        "predicted_occluder":
            predicted_occluder,

        "selected_candidate":
            selected_candidate,

        "candidates":
            candidates
    }