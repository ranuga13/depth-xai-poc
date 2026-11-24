import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from src.datasets.kins_loader import KINSLoader
from src.roles.scene_roles import find_foreground_occluder
from src.roles.depth_role_inference import (
    infer_scene_roles_from_depth
)


# ==========================================================
# Configuration
# ==========================================================

ANNOTATION_PATH = (
    "data/annotations/update_train_2020.json"
)

IMAGE_DIR = (
    "data/images/training/image_2"
)

VALIDATED_RESULTS_PATH = (
    "results/batch_results_validated.csv"
)

DEPTH_DIR = Path(
    "results/depth_batch"
)

OUTPUT_DIR = Path(
    "results/automatic_role_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# Setup
# ==========================================================

loader = KINSLoader(
    ANNOTATION_PATH
)

validated_df = pd.read_csv(
    VALIDATED_RESULTS_PATH
)

TARGET_IDS = sorted(
    validated_df[
        "target_ann_id"
    ].unique().tolist()
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
# Process all targets
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
            "Target annotation not found."
        )

        continue


    # ======================================================
    # Image
    # ======================================================

    image_info = loader.get_image(
        target_ann[
            "image_id"
        ]
    )

    if image_info is None:

        print(
            "Image information not found."
        )

        continue


    file_name = image_info[
        "file_name"
    ]


    image_path = (
        Path(IMAGE_DIR)
        / file_name
    )


    image = cv2.imread(
        str(image_path)
    )


    if image is None:

        print(
            f"Could not load image: "
            f"{image_path}"
        )

        continue


    height, width = image.shape[:2]


    # ======================================================
    # Load depth map
    # ======================================================

    depth_path = (
        DEPTH_DIR
        / f"target_{target_id}_depth.npy"
    )


    if not depth_path.exists():

        print(
            "Depth map not found."
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
            (width, height),
            interpolation=cv2.INTER_CUBIC
        )


    # ======================================================
    # Image annotations
    # ======================================================

    image_annotations = (
        loader.get_annotations_for_image(
            target_ann[
                "image_id"
            ]
        )
    )


    # ======================================================
    # Automatic role inference
    #
    # IMPORTANT:
    # This does NOT use oco_id / ico_id.
    # ======================================================

    roles = (
        infer_scene_roles_from_depth(
            target_annotation=target_ann,
            image_annotations=image_annotations,
            depth_map=depth_map,
            height=height,
            width=width
        )
    )


    predicted_occluder = (
        roles[
            "predicted_occluder"
        ]
    )


    # ======================================================
    # Predicted occluder information
    # ======================================================

    if predicted_occluder is not None:

        predicted_id = (
            predicted_occluder[
                "id"
            ]
        )

        predicted_class = (
            loader.get_category_name(
                predicted_occluder[
                    "category_id"
                ]
            )
        )

    else:

        predicted_id = None
        predicted_class = None


    # ======================================================
    # KINS reference occluder
    #
    # ONLY used for evaluation after prediction.
    # ======================================================

    kins_occluder = (
        find_foreground_occluder(
            target_annotation=target_ann,
            image_annotations=image_annotations,
            height=height,
            width=width
        )
    )


    if kins_occluder is not None:

        kins_id = (
            kins_occluder[
                "id"
            ]
        )

        kins_class = (
            loader.get_category_name(
                kins_occluder[
                    "category_id"
                ]
            )
        )

    else:

        kins_id = None
        kins_class = None


    # ======================================================
    # Determine result status
    # ======================================================

    if predicted_id is None:

        status = (
            "no_prediction"
        )

    elif predicted_id == kins_id:

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

    selected_overlap = np.nan
    selected_depth = np.nan
    selected_depth_difference = np.nan
    selected_score = np.nan


    if predicted_id is not None:

        for candidate in roles[
            "candidates"
        ]:

            if (
                candidate[
                    "annotation_id"
                ]
                == predicted_id
            ):

                selected_overlap = (
                    candidate[
                        "overlap_ratio"
                    ]
                )

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
        f"KINS reference: "
        f"{kins_id} "
        f"({kins_class})"
    )

    print(
        f"Predicted: "
        f"{predicted_id} "
        f"({predicted_class})"
    )

    print(
        f"Target depth: "
        f"{roles['target_depth']:.4f}"
    )


    if predicted_id is not None:

        print(
            f"Predicted depth: "
            f"{selected_depth:.4f}"
        )

        print(
            f"Depth difference: "
            f"{selected_depth_difference:.4f}"
        )

        print(
            f"Overlap ratio: "
            f"{selected_overlap:.4f}"
        )

        print(
            f"Selection score: "
            f"{selected_score:.4f}"
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

            "kins_occluder_id":
                kins_id,

            "kins_occluder_class":
                kins_class,

            "predicted_occluder_id":
                predicted_id,

            "predicted_occluder_class":
                predicted_class,

            "target_depth":
                roles[
                    "target_depth"
                ],

            "predicted_occluder_depth":
                selected_depth,

            "depth_difference":
                selected_depth_difference,

            "bbox_overlap_ratio":
                selected_overlap,

            "selection_score":
                selected_score,

            "num_candidates":
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

            "correct":
                (
                    predicted_id
                    == kins_id
                )
        }
    )


# ==========================================================
# Save detailed results
# ==========================================================

result_df = pd.DataFrame(
    rows
)


output_csv = (
    OUTPUT_DIR
    / "automatic_role_validation.csv"
)


result_df.to_csv(
    output_csv,
    index=False
)


# ==========================================================
# Summary
# ==========================================================

total = len(
    result_df
)


correct_count = int(
    (
        result_df[
            "status"
        ]
        == "correct"
    ).sum()
)


wrong_count = int(
    (
        result_df[
            "status"
        ]
        == "wrong"
    ).sum()
)


no_prediction_count = int(
    (
        result_df[
            "status"
        ]
        == "no_prediction"
    ).sum()
)


if total > 0:

    accuracy = (
        correct_count
        / total
        * 100
    )

else:

    accuracy = 0.0


predicted_count = (
    correct_count
    + wrong_count
)


if predicted_count > 0:

    conditional_accuracy = (
        correct_count
        / predicted_count
        * 100
    )

else:

    conditional_accuracy = 0.0


print()
print("=" * 90)
print("AUTOMATIC DEPTH-BASED ROLE VALIDATION SUMMARY")
print("=" * 90)

print(
    f"Total targets: "
    f"{total}"
)

print(
    f"Correct: "
    f"{correct_count}"
)

print(
    f"Wrong: "
    f"{wrong_count}"
)

print(
    f"No prediction: "
    f"{no_prediction_count}"
)

print(
    f"Overall accuracy: "
    f"{accuracy:.2f}%"
)

print(
    f"Accuracy when prediction exists: "
    f"{conditional_accuracy:.2f}%"
)

print()
print(
    f"Saved results to: "
    f"{output_csv}"
)


# ==========================================================
# Show failures
# ==========================================================

failure_df = result_df[
    result_df[
        "status"
    ] != "correct"
]


if not failure_df.empty:

    print()
    print("=" * 90)
    print("FAILED CASES")
    print("=" * 90)

    print(
        failure_df[
            [
                "target_ann_id",
                "kins_occluder_id",
                "predicted_occluder_id",
                "target_depth",
                "predicted_occluder_depth",
                "depth_difference",
                "bbox_overlap_ratio",
                "status"
            ]
        ].to_string(
            index=False
        )
    )