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
    infer_roles_from_predicted_instances
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

VALIDATED_RESULTS_PATH = Path(
    "results/batch_results_validated.csv"
)

DEPTH_DIR = Path(
    "results/depth_batch"
)

OUTPUT_DIR = Path(
    "results/predicted_role_validation"
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


# Evaluation thresholds only
OCCLUDER_BBOX_IOU_THRESHOLD = (
    0.50
)

OCCLUDER_MASK_IOU_THRESHOLD = (
    0.50
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


validated_df = pd.read_csv(
    VALIDATED_RESULTS_PATH
)


TARGET_IDS = sorted(
    validated_df[
        "target_ann_id"
    ]
    .unique()
    .tolist()
)


print(
    f"Validated targets: "
    f"{len(TARGET_IDS)}"
)


# ==========================================================
# Results
# ==========================================================

rows = []


# ==========================================================
# Process targets
# ==========================================================

for index, target_id in enumerate(
    TARGET_IDS
):

    print()
    print("=" * 90)

    print(
        f"TARGET "
        f"{index + 1}/"
        f"{len(TARGET_IDS)} "
        f"| ID {target_id}"
    )

    print("=" * 90)


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
            "Target annotation missing."
        )

        rows.append(
            {
                "target_ann_id":
                    target_id,

                "status":
                    "target_annotation_missing"
            }
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
            "Image metadata missing."
        )

        rows.append(
            {
                "target_ann_id":
                    target_id,

                "status":
                    "image_metadata_missing"
            }
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
            "Image could not be loaded."
        )

        rows.append(
            {
                "target_ann_id":
                    target_id,

                "file_name":
                    file_name,

                "status":
                    "image_load_failed"
            }
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


    original_instance_count = (
        len(raw_instances)
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


    print(
        f"Instances: "
        f"{original_instance_count} "
        f"-> {len(instances)} "
        f"after deduplication"
    )


    # ======================================================
    # Match target to predicted instance
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
            f"Target segmentation match failed. "
            f"Best IoU: "
            f"{target_match_iou:.4f}"
        )


        rows.append(
            {
                "target_ann_id":
                    target_id,

                "image_id":
                    target_ann[
                        "image_id"
                    ],

                "file_name":
                    file_name,

                "target_class":
                    target_class,

                "target_match_iou":
                    target_match_iou,

                "original_instances":
                    original_instance_count,

                "deduplicated_instances":
                    len(
                        instances
                    ),

                "duplicates_removed":
                    len(
                        removed_duplicates
                    ),

                "status":
                    "target_match_failed",

                "physical_occluder_correct":
                    False
            }
        )

        continue


    # ======================================================
    # Depth
    # ======================================================

    depth_path = (
        DEPTH_DIR
        / f"target_{target_id}_depth.npy"
    )


    if not depth_path.exists():

        print(
            "Depth map missing."
        )

        rows.append(
            {
                "target_ann_id":
                    target_id,

                "image_id":
                    target_ann[
                        "image_id"
                    ],

                "file_name":
                    file_name,

                "target_class":
                    target_class,

                "target_match_iou":
                    target_match_iou,

                "status":
                    "depth_missing",

                "physical_occluder_correct":
                    False
            }
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
    # Fully predicted role inference
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
    # KINS reference occluder
    #
    # Evaluation only
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

        rows.append(
            {
                "target_ann_id":
                    target_id,

                "image_id":
                    target_ann[
                        "image_id"
                    ],

                "file_name":
                    file_name,

                "target_class":
                    target_class,

                "target_match_iou":
                    target_match_iou,

                "status":
                    "no_kins_occluder",

                "physical_occluder_correct":
                    False
            }
        )

        continue


    kins_occluder_id = (
        kins_occluder[
            "id"
        ]
    )


    kins_occluder_class = (
        loader.get_category_name(
            kins_occluder[
                "category_id"
            ]
        )
    )


    kins_occluder_box = (
        xywh_to_xyxy(
            kins_occluder[
                "i_bbox"
            ]
        )
    )


    kins_occluder_mask = (
        polygon_to_mask(
            kins_occluder[
                "i_segm"
            ],
            height,
            width
        )
    )


    # ======================================================
    # No predicted occluder
    # ======================================================

    if predicted_occluder is None:

        print(
            "Predicted occluder: None"
        )


        rows.append(
            {
                "target_ann_id":
                    target_id,

                "image_id":
                    target_ann[
                        "image_id"
                    ],

                "file_name":
                    file_name,

                "target_class":
                    target_class,

                "target_instance_id":
                    target_instance[
                        "instance_id"
                    ],

                "target_segmentation_confidence":
                    target_instance[
                        "confidence"
                    ],

                "target_match_iou":
                    target_match_iou,

                "target_depth":
                    roles[
                        "target_depth"
                    ],

                "kins_occluder_id":
                    kins_occluder_id,

                "kins_occluder_class":
                    kins_occluder_class,

                "predicted_occluder_instance_id":
                    None,

                "predicted_occluder_class":
                    None,

                "predicted_occluder_confidence":
                    None,

                "occluder_bbox_iou":
                    0.0,

                "occluder_mask_iou":
                    0.0,

                "original_instances":
                    original_instance_count,

                "deduplicated_instances":
                    len(
                        instances
                    ),

                "duplicates_removed":
                    len(
                        removed_duplicates
                    ),

                "num_role_candidates":
                    len(
                        roles[
                            "candidates"
                        ]
                    ),

                "status":
                    "no_prediction",

                "physical_occluder_correct":
                    False
            }
        )

        continue


    # ======================================================
    # Predicted occluder geometry comparison
    # ======================================================

    predicted_box = (
        predicted_occluder[
            "bbox"
        ]
    )


    predicted_mask = (
        predicted_occluder[
            "mask"
        ]
    )


    bbox_iou = (
        calculate_iou(
            predicted_box,
            kins_occluder_box
        )
    )


    mask_iou = (
        calculate_mask_iou(
            predicted_mask,
            kins_occluder_mask
        )
    )


    physical_correct = (
        bbox_iou
        >= OCCLUDER_BBOX_IOU_THRESHOLD

        or

        mask_iou
        >= OCCLUDER_MASK_IOU_THRESHOLD
    )


    if physical_correct:

        status = (
            "correct"
        )

    else:

        status = (
            "wrong"
        )


    # ======================================================
    # Selected candidate details
    # ======================================================

    selected_depth = np.nan
    selected_depth_difference = np.nan
    selected_overlap = np.nan
    selected_score = np.nan


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
            predicted_occluder[
                "instance_id"
            ]
        ):

            selected_depth = (
                candidate[
                    "candidate_depth"
                ]
            )

            selected_depth_difference = (
                candidate[
                    "depth_difference"
                ]
            )

            selected_overlap = (
                candidate[
                    "overlap_ratio"
                ]
            )

            selected_score = (
                candidate[
                    "score"
                ]
            )

            break


    # ======================================================
    # Console output
    # ======================================================

    print(
        f"Target match IoU: "
        f"{target_match_iou:.4f}"
    )


    print(
        f"KINS occluder: "
        f"{kins_occluder_id} "
        f"({kins_occluder_class})"
    )


    print(
        f"Predicted instance: "
        f"{predicted_occluder['instance_id']} "
        f"({predicted_occluder['class_name']})"
    )


    print(
        f"BBox IoU: "
        f"{bbox_iou:.4f}"
    )


    print(
        f"Mask IoU: "
        f"{mask_iou:.4f}"
    )


    print(
        f"Status: "
        f"{status}"
    )


    # ======================================================
    # Save row
    # ======================================================

    rows.append(
        {
            "target_ann_id":
                target_id,

            "image_id":
                target_ann[
                    "image_id"
                ],

            "file_name":
                file_name,

            "target_class":
                target_class,

            "target_instance_id":
                target_instance[
                    "instance_id"
                ],

            "target_segmentation_class":
                target_instance[
                    "class_name"
                ],

            "target_segmentation_confidence":
                target_instance[
                    "confidence"
                ],

            "target_match_iou":
                target_match_iou,

            "target_depth":
                roles[
                    "target_depth"
                ],

            "kins_occluder_id":
                kins_occluder_id,

            "kins_occluder_class":
                kins_occluder_class,

            "predicted_occluder_instance_id":
                predicted_occluder[
                    "instance_id"
                ],

            "predicted_occluder_class":
                predicted_occluder[
                    "class_name"
                ],

            "predicted_occluder_confidence":
                predicted_occluder[
                    "confidence"
                ],

            "predicted_occluder_depth":
                selected_depth,

            "depth_difference":
                selected_depth_difference,

            "bbox_overlap_ratio":
                selected_overlap,

            "selection_score":
                selected_score,

            "occluder_bbox_iou":
                bbox_iou,

            "occluder_mask_iou":
                mask_iou,

            "original_instances":
                original_instance_count,

            "deduplicated_instances":
                len(
                    instances
                ),

            "duplicates_removed":
                len(
                    removed_duplicates
                ),

            "num_role_candidates":
                len(
                    roles[
                        "candidates"
                    ]
                ),

            "target_pixels":
                int(
                    roles[
                        "target_mask"
                    ].sum()
                ),

            "occluder_pixels":
                int(
                    roles[
                        "occluder_mask"
                    ].sum()
                ),

            "context_pixels":
                int(
                    roles[
                        "context_mask"
                    ].sum()
                ),

            "status":
                status,

            "physical_occluder_correct":
                physical_correct
        }
    )


# ==========================================================
# Save results
# ==========================================================

result_df = pd.DataFrame(
    rows
)


output_path = (
    OUTPUT_DIR
    / "predicted_role_validation.csv"
)


result_df.to_csv(
    output_path,
    index=False
)


# ==========================================================
# Summary
# ==========================================================

total = len(
    result_df
)


correct = int(
    (
        result_df[
            "status"
        ]
        == "correct"
    ).sum()
)


wrong = int(
    (
        result_df[
            "status"
        ]
        == "wrong"
    ).sum()
)


no_prediction = int(
    (
        result_df[
            "status"
        ]
        == "no_prediction"
    ).sum()
)


target_match_failed = int(
    (
        result_df[
            "status"
        ]
        == "target_match_failed"
    ).sum()
)


other_failures = (
    total
    - correct
    - wrong
    - no_prediction
    - target_match_failed
)


overall_accuracy = (
    correct
    / total
    * 100
    if total > 0
    else 0.0
)


successful_target_matches = (
    total
    - target_match_failed
)


if successful_target_matches > 0:

    accuracy_after_target_match = (
        correct
        / successful_target_matches
        * 100
    )

else:

    accuracy_after_target_match = (
        0.0
    )


predictions_exist = (
    correct
    + wrong
)


if predictions_exist > 0:

    accuracy_when_prediction_exists = (
        correct
        / predictions_exist
        * 100
    )

else:

    accuracy_when_prediction_exists = (
        0.0
    )


print()
print("=" * 90)
print("FULLY PREDICTED ROLE VALIDATION SUMMARY")
print("=" * 90)


print(
    f"Total targets: "
    f"{total}"
)

print(
    f"Correct physical occluder: "
    f"{correct}"
)

print(
    f"Wrong physical occluder: "
    f"{wrong}"
)

print(
    f"No occluder prediction: "
    f"{no_prediction}"
)

print(
    f"Target segmentation match failed: "
    f"{target_match_failed}"
)

print(
    f"Other failures: "
    f"{other_failures}"
)


print()

print(
    f"Overall accuracy: "
    f"{overall_accuracy:.2f}%"
)

print(
    f"Accuracy after successful target match: "
    f"{accuracy_after_target_match:.2f}%"
)

print(
    f"Accuracy when occluder prediction exists: "
    f"{accuracy_when_prediction_exists:.2f}%"
)


# ==========================================================
# Geometry summary
# ==========================================================

valid_predictions = result_df[
    result_df[
        "status"
    ].isin(
        [
            "correct",
            "wrong"
        ]
    )
]


if not valid_predictions.empty:

    print()
    print("=" * 90)
    print("GEOMETRY SUMMARY")
    print("=" * 90)

    print(
        f"Mean target match IoU: "
        f"{valid_predictions['target_match_iou'].mean():.4f}"
    )

    print(
        f"Mean occluder bbox IoU: "
        f"{valid_predictions['occluder_bbox_iou'].mean():.4f}"
    )

    print(
        f"Median occluder bbox IoU: "
        f"{valid_predictions['occluder_bbox_iou'].median():.4f}"
    )

    print(
        f"Mean occluder mask IoU: "
        f"{valid_predictions['occluder_mask_iou'].mean():.4f}"
    )

    print(
        f"Median occluder mask IoU: "
        f"{valid_predictions['occluder_mask_iou'].median():.4f}"
    )


# ==========================================================
# Failures
# ==========================================================

failures = result_df[
    result_df[
        "status"
    ]
    != "correct"
]


if not failures.empty:

    print()
    print("=" * 90)
    print("FAILED CASES")
    print("=" * 90)

    columns = [
        column
        for column in [
            "target_ann_id",
            "target_class",
            "target_match_iou",
            "kins_occluder_id",
            "kins_occluder_class",
            "predicted_occluder_instance_id",
            "predicted_occluder_class",
            "occluder_bbox_iou",
            "occluder_mask_iou",
            "status"
        ]
        if column in failures.columns
    ]

    print(
        failures[
            columns
        ].to_string(
            index=False
        )
    )


print()
print(
    f"Saved to: "
    f"{output_path}"
)