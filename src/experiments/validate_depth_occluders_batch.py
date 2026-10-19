import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from src.datasets.kins_loader import KINSLoader
from src.depth.depth_estimator import DepthEstimator
from src.roles.scene_roles import find_foreground_occluder
from src.roles.depth_occluder import (
    predict_foreground_occluder_from_depth
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
    "results/depth_occluder_validation"
)

DEPTH_DIR.mkdir(
    parents=True,
    exist_ok=True
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

depth_estimator = DepthEstimator()


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
# Process targets
# ==========================================================

for index, target_id in enumerate(
    TARGET_IDS
):

    print()
    print("=" * 80)
    print(
        f"TARGET {index + 1}/{len(TARGET_IDS)} "
        f"| ID {target_id}"
    )
    print("=" * 80)


    target_ann = loader.get_annotation_by_id(
        target_id
    )

    if target_ann is None:

        print(
            "Target annotation not found."
        )

        continue


    image_info = loader.get_image(
        target_ann["image_id"]
    )

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
    # Depth
    # ======================================================

    depth_path = (
        DEPTH_DIR
        / f"target_{target_id}_depth.npy"
    )


    if depth_path.exists():

        depth_map = np.load(
            depth_path
        )

        print(
            "Loaded cached depth map."
        )

    else:

        depth_map = depth_estimator.predict(
            image
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


        np.save(
            depth_path,
            depth_map
        )

        print(
            "Generated depth map."
        )


    # ======================================================
    # Image annotations
    # ======================================================

    image_annotations = (
        loader.get_annotations_for_image(
            target_ann["image_id"]
        )
    )


    # ======================================================
    # KINS occluder
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
    # Depth prediction
    # ======================================================

    prediction = (
        predict_foreground_occluder_from_depth(
            target_annotation=target_ann,
            image_annotations=image_annotations,
            depth_map=depth_map,
            height=height,
            width=width
        )
    )


    predicted_occluder = (
        prediction[
            "predicted_occluder"
        ]
    )


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
    # Match result
    # ======================================================

    correct = (
        kins_id
        == predicted_id
    )


    if predicted_occluder is None:

        prediction_status = (
            "no_prediction"
        )

    elif correct:

        prediction_status = (
            "correct"
        )

    else:

        prediction_status = (
            "wrong"
        )


    # ======================================================
    # Predicted candidate depth
    # ======================================================

    predicted_depth = np.nan
    predicted_depth_difference = np.nan
    predicted_overlap = np.nan


    if predicted_id is not None:

        for candidate in prediction[
            "candidates"
        ]:

            if (
                candidate[
                    "annotation_id"
                ]
                == predicted_id
            ):

                predicted_depth = (
                    candidate[
                        "candidate_depth"
                    ]
                )

                predicted_depth_difference = (
                    candidate[
                        "depth_difference"
                    ]
                )

                predicted_overlap = (
                    candidate[
                        "overlap_pixels"
                    ]
                )

                break


    print(
        f"KINS occluder: "
        f"{kins_id} ({kins_class})"
    )

    print(
        f"Predicted: "
        f"{predicted_id} ({predicted_class})"
    )

    print(
        f"Status: "
        f"{prediction_status}"
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

            "kins_occluder_id":
                kins_id,

            "kins_occluder_class":
                kins_class,

            "predicted_occluder_id":
                predicted_id,

            "predicted_occluder_class":
                predicted_class,

            "target_median_depth":
                prediction[
                    "target_depth"
                ],

            "predicted_occluder_depth":
                predicted_depth,

            "depth_difference":
                predicted_depth_difference,

            "overlap_pixels":
                predicted_overlap,

            "prediction_status":
                prediction_status,

            "correct":
                correct
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
    / "depth_occluder_validation.csv"
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
            "prediction_status"
        ]
        == "correct"
    ).sum()
)

wrong_count = int(
    (
        result_df[
            "prediction_status"
        ]
        == "wrong"
    ).sum()
)

no_prediction_count = int(
    (
        result_df[
            "prediction_status"
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


print()
print("=" * 80)
print("DEPTH OCCLUDER VALIDATION SUMMARY")
print("=" * 80)

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
    f"Accuracy: "
    f"{accuracy:.2f}%"
)

print()
print(
    f"Saved results to: "
    f"{output_csv}"
)