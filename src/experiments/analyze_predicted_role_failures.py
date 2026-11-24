import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from src.datasets.kins_loader import KINSLoader

from src.detector.yolo_segmenter import (
    YOLOSegmenter
)

from src.detector.segmentation_matcher import (
    match_target_to_segment_instance
)

from src.detector.instance_deduplicator import (
    deduplicate_instances,
    calculate_mask_iou
)

from src.detector.target_matcher import (
    calculate_iou,
    xywh_to_xyxy
)

from src.roles.predicted_role_inference import (
    infer_roles_from_predicted_instances,
    get_median_depth,
    bbox_intersection_ratio,
    normalized_bbox_gap
)

from src.roles.scene_roles import (
    find_foreground_occluder,
    polygon_to_mask
)


# ==========================================================
# Configuration
# ==========================================================

ANNOTATION_PATH = (
    "data/annotations/update_train_2020.json"
)

IMAGE_DIR = Path(
    "data/images/training/image_2"
)

DEPTH_DIR = Path(
    "results/depth_batch"
)

VALIDATION_PATH = Path(
    "results/predicted_role_validation/"
    "predicted_role_validation.csv"
)

OUTPUT_DIR = Path(
    "results/predicted_role_failure_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SEGMENTATION_MODEL = (
    "yolo11n-seg.pt"
)

TARGET_MATCH_IOU_THRESHOLD = (
    0.30
)

DEDUP_BBOX_IOU_THRESHOLD = (
    0.80
)

DEDUP_MASK_IOU_THRESHOLD = (
    0.70
)


# ==========================================================
# Setup
# ==========================================================

loader = KINSLoader(
    ANNOTATION_PATH
)

segmenter = YOLOSegmenter(
    SEGMENTATION_MODEL
)

validation_df = pd.read_csv(
    VALIDATION_PATH
)

failure_df = validation_df[
    validation_df["status"] != "correct"
]

failure_ids = (
    failure_df[
        "target_ann_id"
    ]
    .astype(int)
    .tolist()
)


print()
print("=" * 110)
print("PREDICTED ROLE FAILURE ANALYSIS")
print("=" * 110)

print(
    f"Failure targets: "
    f"{failure_ids}"
)


rows = []


# ==========================================================
# Analyze each failure
# ==========================================================

for target_id in failure_ids:

    print()
    print("=" * 110)
    print(
        f"TARGET {target_id}"
    )
    print("=" * 110)


    # ======================================================
    # Target annotation
    # ======================================================

    target_ann = (
        loader.get_annotation_by_id(
            target_id
        )
    )

    if target_ann is None:

        print(
            "Target annotation not found."
        )

        continue


    target_class = (
        loader.get_category_name(
            target_ann[
                "category_id"
            ]
        )
    )


    # ======================================================
    # Image
    # ======================================================

    image_info = (
        loader.get_image(
            target_ann[
                "image_id"
            ]
        )
    )

    if image_info is None:

        print(
            "Image information not found."
        )

        continue


    file_name = (
        image_info[
            "file_name"
        ]
    )

    image_path = (
        IMAGE_DIR
        / file_name
    )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            "Image load failed."
        )

        continue


    height, width = (
        image.shape[:2]
    )


    # ======================================================
    # YOLO11n-seg
    # ======================================================

    raw_instances = (
        segmenter.segment(
            image
        )
    )

    instances, removed_duplicates = (
        deduplicate_instances(
            raw_instances,

            bbox_iou_threshold=
                DEDUP_BBOX_IOU_THRESHOLD,

            mask_iou_threshold=
                DEDUP_MASK_IOU_THRESHOLD
        )
    )


    # ======================================================
    # Match target
    # ======================================================

    target_instance, target_match_iou = (
        match_target_to_segment_instance(
            target_annotation=
                target_ann,

            target_class=
                target_class,

            instances=
                instances,

            min_iou=
                TARGET_MATCH_IOU_THRESHOLD
        )
    )

    if target_instance is None:

        print(
            "Target segmentation match failed."
        )

        print(
            f"Best IoU: "
            f"{target_match_iou:.4f}"
        )

        continue


    # ======================================================
    # Load depth
    # ======================================================

    depth_path = (
        DEPTH_DIR
        / f"target_{target_id}_depth.npy"
    )

    if not depth_path.exists():

        print(
            "Depth map missing."
        )

        continue


    depth_map = np.load(
        depth_path
    )

    if depth_map.shape != (
        height,
        width
    ):

        depth_map = cv2.resize(
            depth_map,
            (
                width,
                height
            ),
            interpolation=
                cv2.INTER_CUBIC
        )


    # ======================================================
    # KINS reference occluder
    # ======================================================

    image_annotations = (
        loader.get_annotations_for_image(
            target_ann[
                "image_id"
            ]
        )
    )

    kins_occluder = (
        find_foreground_occluder(
            target_annotation=
                target_ann,

            image_annotations=
                image_annotations,

            height=
                height,

            width=
                width
        )
    )

    if kins_occluder is None:

        print(
            "No KINS reference occluder."
        )

        continue


    kins_box = (
        xywh_to_xyxy(
            kins_occluder[
                "i_bbox"
            ]
        )
    )

    kins_mask = (
        polygon_to_mask(
            kins_occluder[
                "i_segm"
            ],
            height,
            width
        )
    )

    kins_class = (
        loader.get_category_name(
            kins_occluder[
                "category_id"
            ]
        )
    )


    # ======================================================
    # Predicted role inference
    # ======================================================

    roles = (
        infer_roles_from_predicted_instances(
            target_instance=
                target_instance,

            instances=
                instances,

            depth_map=
                depth_map,

            height=
                height,

            width=
                width
        )
    )

    predicted_occluder = (
        roles[
            "predicted_occluder"
        ]
    )


    # ======================================================
    # Main target information
    # ======================================================

    print(
        f"Target class: "
        f"{target_class}"
    )

    print(
        f"Target match IoU: "
        f"{target_match_iou:.4f}"
    )

    print(
        f"Target depth: "
        f"{roles['target_depth']:.4f}"
    )

    print(
        f"KINS occluder: "
        f"{kins_occluder['id']} "
        f"({kins_class})"
    )

    if predicted_occluder is None:

        print(
            "Predicted occluder: None"
        )

    else:

        print(
            f"Predicted occluder: "
            f"instance "
            f"{predicted_occluder['instance_id']} "
            f"({predicted_occluder['class_name']})"
        )


    # ======================================================
    # ALL predicted instances
    #
    # This includes candidates rejected by role inference.
    # This is where we inspect normalized gap directly.
    # ======================================================

    print()
    print(
        f"{'Inst':>6}"
        f"{'Class':>12}"
        f"{'Conf':>9}"
        f"{'TargetOv':>11}"
        f"{'Gap':>10}"
        f"{'Depth':>10}"
        f"{'DepthDiff':>11}"
        f"{'KinsBBox':>11}"
        f"{'KinsMask':>11}"
        f"{'Closer':>9}"
    )

    print(
        "-" * 110
    )


    best_kins_match = None
    best_kins_score = -1.0


    for instance in instances:

        if (
            instance[
                "instance_id"
            ]
            ==
            target_instance[
                "instance_id"
            ]
        ):
            continue


        # --------------------------------------------------
        # Original target-box overlap
        # --------------------------------------------------

        target_overlap = (
            bbox_intersection_ratio(
                target_instance[
                    "bbox"
                ],

                instance[
                    "bbox"
                ]
            )
        )


        # --------------------------------------------------
        # Normalized gap
        # --------------------------------------------------

        gap = (
            normalized_bbox_gap(
                target_instance[
                    "bbox"
                ],

                instance[
                    "bbox"
                ]
            )
        )


        # --------------------------------------------------
        # Depth
        # --------------------------------------------------

        candidate_depth = (
            get_median_depth(
                instance[
                    "mask"
                ],
                depth_map
            )
        )

        depth_difference = (
            candidate_depth
            -
            roles[
                "target_depth"
            ]
        )


        # --------------------------------------------------
        # Geometry relative to KINS occluder
        # --------------------------------------------------

        bbox_iou_to_kins = (
            calculate_iou(
                instance[
                    "bbox"
                ],
                kins_box
            )
        )

        mask_iou_to_kins = (
            calculate_mask_iou(
                instance[
                    "mask"
                ],
                kins_mask
            )
        )

        closer = (
            depth_difference
            > 0.05
        )

        geometry_match_score = max(
            bbox_iou_to_kins,
            mask_iou_to_kins
        )

        if (
            geometry_match_score
            > best_kins_score
        ):

            best_kins_score = (
                geometry_match_score
            )

            best_kins_match = (
                instance
            )


        print(
            f"{instance['instance_id']:>6}"
            f"{instance['class_name']:>12}"
            f"{instance['confidence']:>9.4f}"
            f"{target_overlap:>11.4f}"
            f"{gap:>10.4f}"
            f"{candidate_depth:>10.4f}"
            f"{depth_difference:>11.4f}"
            f"{bbox_iou_to_kins:>11.4f}"
            f"{mask_iou_to_kins:>11.4f}"
            f"{str(closer):>9}"
        )


    # ======================================================
    # Current role-inference candidate scores
    # ======================================================

    print()
    print("=" * 110)
    print("ROLE-INFERENCE CANDIDATE SCORES")
    print("=" * 110)

    print(
        f"{'Inst':>6}"
        f"{'Class':>12}"
        f"{'Conf':>9}"
        f"{'ExpOv':>10}"
        f"{'Gap':>10}"
        f"{'GapProx':>10}"
        f"{'Proximity':>11}"
        f"{'DepthDiff':>11}"
        f"{'Score':>12}"
        f"{'Selected':>10}"
    )

    print(
        "-" * 110
    )


    for candidate in (
        roles[
            "candidates"
        ]
    ):

        is_selected = (
            predicted_occluder
            is not None
            and
            candidate[
                "instance_id"
            ]
            ==
            predicted_occluder[
                "instance_id"
            ]
        )


        print(
            f"{candidate['instance_id']:>6}"
            f"{candidate['class_name']:>12}"
            f"{candidate['confidence']:>9.4f}"
            f"{candidate['overlap_ratio']:>10.4f}"
            f"{candidate['normalized_gap']:>10.4f}"
            f"{candidate['gap_proximity']:>10.4f}"
            f"{candidate['proximity_score']:>11.4f}"
            f"{candidate['depth_difference']:>11.4f}"
            f"{candidate['score']:>12.4f}"
            f"{str(is_selected):>10}"
        )


        instance = (
            candidate[
                "instance"
            ]
        )

        bbox_iou_to_kins = (
            calculate_iou(
                instance[
                    "bbox"
                ],
                kins_box
            )
        )

        mask_iou_to_kins = (
            calculate_mask_iou(
                instance[
                    "mask"
                ],
                kins_mask
            )
        )


        rows.append(
            {
                "target_ann_id":
                    target_id,

                "candidate_instance_id":
                    candidate[
                        "instance_id"
                    ],

                "candidate_class":
                    candidate[
                        "class_name"
                    ],

                "candidate_confidence":
                    candidate[
                        "confidence"
                    ],

                "expanded_overlap_ratio":
                    candidate[
                        "overlap_ratio"
                    ],

                "normalized_gap":
                    candidate[
                        "normalized_gap"
                    ],

                "gap_proximity":
                    candidate[
                        "gap_proximity"
                    ],

                "proximity_score":
                    candidate[
                        "proximity_score"
                    ],

                "candidate_depth":
                    candidate[
                        "candidate_depth"
                    ],

                "depth_difference":
                    candidate[
                        "depth_difference"
                    ],

                "is_closer":
                    candidate[
                        "is_closer"
                    ],

                "geometry_valid":
                    candidate[
                        "geometry_valid"
                    ],

                "final_score":
                    candidate[
                        "score"
                    ],

                "bbox_iou_to_kins_occluder":
                    bbox_iou_to_kins,

                "mask_iou_to_kins_occluder":
                    mask_iou_to_kins,

                "was_selected":
                    is_selected
            }
        )


    # ======================================================
    # Best predicted instance matching true KINS occluder
    # ======================================================

    print()

    if best_kins_match is not None:

        best_depth = (
            get_median_depth(
                best_kins_match[
                    "mask"
                ],
                depth_map
            )
        )

        best_depth_diff = (
            best_depth
            -
            roles[
                "target_depth"
            ]
        )

        best_gap = (
            normalized_bbox_gap(
                target_instance[
                    "bbox"
                ],

                best_kins_match[
                    "bbox"
                ]
            )
        )


        print(
            "Best predicted instance matching "
            "the true KINS occluder:"
        )

        print(
            f"  Instance: "
            f"{best_kins_match['instance_id']}"
        )

        print(
            f"  Class: "
            f"{best_kins_match['class_name']}"
        )

        print(
            f"  Confidence: "
            f"{best_kins_match['confidence']:.4f}"
        )

        print(
            f"  Normalized gap: "
            f"{best_gap:.4f}"
        )

        print(
            f"  Depth difference: "
            f"{best_depth_diff:.4f}"
        )

        print(
            f"  Would pass depth threshold: "
            f"{best_depth_diff > 0.05}"
        )


        matching_role_candidate = None

        for candidate in (
            roles[
                "candidates"
            ]
        ):

            if (
                candidate[
                    "instance_id"
                ]
                ==
                best_kins_match[
                    "instance_id"
                ]
            ):

                matching_role_candidate = (
                    candidate
                )

                break


        if matching_role_candidate is None:

            print(
                "  Included by current role inference: "
                "False"
            )

        else:

            print(
                "  Included by current role inference: "
                "True"
            )

            print(
                f"  Proximity score: "
                f"{matching_role_candidate['proximity_score']:.4f}"
            )

            print(
                f"  Final selection score: "
                f"{matching_role_candidate['score']:.4f}"
            )


    # ======================================================
    # Selected candidate details
    # ======================================================

    selected_candidate = (
        roles[
            "selected_candidate"
        ]
    )

    if selected_candidate is not None:

        print()
        print(
            "CURRENTLY SELECTED CANDIDATE:"
        )

        print(
            f"  Instance: "
            f"{selected_candidate['instance_id']}"
        )

        print(
            f"  Class: "
            f"{selected_candidate['class_name']}"
        )

        print(
            f"  Confidence: "
            f"{selected_candidate['confidence']:.4f}"
        )

        print(
            f"  Expanded overlap: "
            f"{selected_candidate['overlap_ratio']:.4f}"
        )

        print(
            f"  Normalized gap: "
            f"{selected_candidate['normalized_gap']:.4f}"
        )

        print(
            f"  Proximity score: "
            f"{selected_candidate['proximity_score']:.4f}"
        )

        print(
            f"  Depth difference: "
            f"{selected_candidate['depth_difference']:.4f}"
        )

        print(
            f"  Final score: "
            f"{selected_candidate['score']:.4f}"
        )


# ==========================================================
# Save detailed analysis
# ==========================================================

analysis_df = pd.DataFrame(
    rows
)

output_path = (
    OUTPUT_DIR
    / "failure_candidate_analysis.csv"
)

analysis_df.to_csv(
    output_path,
    index=False
)


# ==========================================================
# Final summary
# ==========================================================

print()
print("=" * 110)
print("FAILURE ANALYSIS COMPLETE")
print("=" * 110)

print(
    f"Saved to: "
    f"{output_path}"
)