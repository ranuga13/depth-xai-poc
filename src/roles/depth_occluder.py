import numpy as np

from src.roles.scene_roles import polygon_to_mask


def get_mask_median_depth(
    mask,
    depth_map
):
    values = depth_map[
        mask
    ]

    if values.size == 0:
        return np.nan

    return float(
        np.median(
            values
        )
    )


def predict_foreground_occluder_from_depth(
    target_annotation,
    image_annotations,
    depth_map,
    height,
    width,
    min_overlap_pixels=10,
    min_depth_difference=0.05
):
    """
    Predict the foreground occluder using:

    1. target/candidate spatial overlap
    2. relative depth
    3. overlap with the hidden part of the target

    For the current Depth Anything V2 output:
    larger raw depth value = closer to camera
    """

    # ======================================================
    # Target masks
    # ======================================================

    target_amodal_mask = polygon_to_mask(
        target_annotation["a_segm"],
        height,
        width
    )

    target_visible_mask = polygon_to_mask(
        target_annotation["i_segm"],
        height,
        width
    )

    target_hidden_mask = (
        target_amodal_mask
        & ~target_visible_mask
    )


    # ======================================================
    # Target depth
    # ======================================================

    target_depth = get_mask_median_depth(
        target_visible_mask,
        depth_map
    )

    if np.isnan(
        target_depth
    ):
        return {
            "predicted_occluder": None,
            "target_depth": np.nan,
            "candidates": []
        }


    candidate_results = []


    # ======================================================
    # Evaluate candidate objects
    # ======================================================

    for candidate in image_annotations:

        if (
            candidate["id"]
            == target_annotation["id"]
        ):
            continue


        candidate_mask = polygon_to_mask(
            candidate["i_segm"],
            height,
            width
        )


        candidate_pixels = int(
            candidate_mask.sum()
        )

        if candidate_pixels == 0:
            continue


        # ----------------------------------------------
        # Does candidate occupy the hidden target area?
        # ----------------------------------------------

        overlap_pixels = int(
            (
                candidate_mask
                & target_hidden_mask
            ).sum()
        )


        if (
            overlap_pixels
            < min_overlap_pixels
        ):
            continue


        # ----------------------------------------------
        # Candidate depth
        # ----------------------------------------------

        candidate_depth = (
            get_mask_median_depth(
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


        # Depth Anything output currently observed:
        # larger = closer
        is_closer = (
            depth_difference
            > min_depth_difference
        )


        candidate_results.append(
            {
                "annotation":
                    candidate,

                "annotation_id":
                    candidate["id"],

                "overlap_pixels":
                    overlap_pixels,

                "candidate_depth":
                    candidate_depth,

                "target_depth":
                    target_depth,

                "depth_difference":
                    depth_difference,

                "is_closer":
                    is_closer
            }
        )


    # ======================================================
    # Keep only candidates physically in front
    # ======================================================

    foreground_candidates = [
        item
        for item in candidate_results
        if item[
            "is_closer"
        ]
    ]


    if not foreground_candidates:

        return {
            "predicted_occluder": None,
            "target_depth": target_depth,
            "candidates": candidate_results
        }


    # ======================================================
    # Select best candidate
    #
    # Primary:
    #   largest overlap with hidden target
    #
    # Secondary:
    #   larger depth advantage
    # ======================================================

    best_candidate = max(
        foreground_candidates,
        key=lambda item: (
            item[
                "overlap_pixels"
            ],
            item[
                "depth_difference"
            ]
        )
    )


    return {
        "predicted_occluder":
            best_candidate[
                "annotation"
            ],

        "target_depth":
            target_depth,

        "candidates":
            candidate_results
    }