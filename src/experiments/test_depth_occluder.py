import cv2
import numpy as np
from pathlib import Path

from src.datasets.kins_loader import KINSLoader
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

TARGET_ID = 68

DEPTH_PATH = (
    "results/depth_single/"
    "target_68_depth.npy"
)


# ==========================================================
# Setup
# ==========================================================

loader = KINSLoader(
    ANNOTATION_PATH
)


target_ann = loader.get_annotation_by_id(
    TARGET_ID
)

if target_ann is None:
    raise ValueError(
        f"Target {TARGET_ID} not found"
    )


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
    raise FileNotFoundError(
        image_path
    )


height, width = image.shape[:2]


# ==========================================================
# Load depth
# ==========================================================

depth_map = np.load(
    DEPTH_PATH
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


# ==========================================================
# Image annotations
# ==========================================================

image_annotations = (
    loader.get_annotations_for_image(
        target_ann["image_id"]
    )
)


# ==========================================================
# KINS ground-truth occluder
# ==========================================================

kins_occluder = (
    find_foreground_occluder(
        target_annotation=target_ann,
        image_annotations=image_annotations,
        height=height,
        width=width
    )
)


# ==========================================================
# Depth-predicted occluder
# ==========================================================

prediction = (
    predict_foreground_occluder_from_depth(
        target_annotation=target_ann,
        image_annotations=image_annotations,
        depth_map=depth_map,
        height=height,
        width=width
    )
)


predicted_occluder = prediction[
    "predicted_occluder"
]


# ==========================================================
# IDs
# ==========================================================

if kins_occluder is not None:

    kins_id = kins_occluder[
        "id"
    ]

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


match = (
    kins_id
    == predicted_id
)


# ==========================================================
# Print summary
# ==========================================================

print()
print("=" * 80)
print("DEPTH-BASED OCCLUDER PREDICTION")
print("=" * 80)

print(
    f"Target ID: {TARGET_ID}"
)

print(
    f"Image: {file_name}"
)

print(
    f"Target median depth: "
    f"{prediction['target_depth']:.4f}"
)

print()

print(
    f"KINS occluder ID: "
    f"{kins_id}"
)

print(
    f"KINS occluder class: "
    f"{kins_class}"
)

print()

print(
    f"Predicted occluder ID: "
    f"{predicted_id}"
)

print(
    f"Predicted occluder class: "
    f"{predicted_class}"
)

print()

print(
    f"Correct prediction: "
    f"{match}"
)


# ==========================================================
# Candidate details
# ==========================================================

print()
print("=" * 80)
print("DEPTH CANDIDATES")
print("=" * 80)


if not prediction[
    "candidates"
]:

    print(
        "No valid candidate objects found."
    )

else:

    print(
        f"{'ID':>8}"
        f"{'Class':>14}"
        f"{'Overlap':>12}"
        f"{'Depth':>12}"
        f"{'Depth Diff':>14}"
        f"{'Closer':>10}"
    )

    print(
        "-" * 72
    )


    for candidate in prediction[
        "candidates"
    ]:

        ann = candidate[
            "annotation"
        ]

        class_name = (
            loader.get_category_name(
                ann[
                    "category_id"
                ]
            )
        )


        print(
            f"{candidate['annotation_id']:>8}"
            f"{class_name:>14}"
            f"{candidate['overlap_pixels']:>12}"
            f"{candidate['candidate_depth']:>12.4f}"
            f"{candidate['depth_difference']:>14.4f}"
            f"{str(candidate['is_closer']):>10}"
        )