import cv2
import numpy as np
from pathlib import Path

from src.datasets.kins_loader import KINSLoader

from src.roles.scene_roles import (
    find_foreground_occluder
)

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

DEPTH_DIR = Path(
    "results/depth_batch"
)

TARGET_ID = 68


# ==========================================================
# Setup
# ==========================================================

loader = KINSLoader(
    ANNOTATION_PATH
)


target_ann = (
    loader.get_annotation_by_id(
        TARGET_ID
    )
)


image_info = loader.get_image(
    target_ann[
        "image_id"
    ]
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
# Load predicted depth
# ==========================================================

depth_path = (
    DEPTH_DIR
    / f"target_{TARGET_ID}_depth.npy"
)


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


# ==========================================================
# Objects in image
# ==========================================================

image_annotations = (
    loader.get_annotations_for_image(
        target_ann[
            "image_id"
        ]
    )
)


# ==========================================================
# Automatic role inference
# ==========================================================

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


if (
    predicted_occluder
    is not None
):

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


# ==========================================================
# KINS occluder
#
# IMPORTANT:
# Only used AFTER prediction for evaluation.
# ==========================================================

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


# ==========================================================
# Results
# ==========================================================

print()
print("=" * 90)
print("AUTOMATIC DEPTH-BASED ROLE INFERENCE")
print("=" * 90)

print(
    f"Target ID: {TARGET_ID}"
)

print(
    f"Image: {file_name}"
)

print(
    f"Target depth: "
    f"{roles['target_depth']:.4f}"
)

print()

print(
    f"Predicted occluder: "
    f"{predicted_id} "
    f"({predicted_class})"
)

print(
    f"KINS reference: "
    f"{kins_id} "
    f"({kins_class})"
)

print(
    f"Correct: "
    f"{predicted_id == kins_id}"
)


# ==========================================================
# Candidates
# ==========================================================

print()
print("=" * 90)
print("CANDIDATES")
print("=" * 90)


print(
    f"{'ID':>8}"
    f"{'Class':>14}"
    f"{'Overlap':>12}"
    f"{'Depth':>12}"
    f"{'Depth Diff':>14}"
    f"{'Closer':>10}"
    f"{'Score':>12}"
)

print(
    "-" * 82
)


for candidate in roles[
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
        f"{candidate['overlap_ratio']:>12.4f}"
        f"{candidate['candidate_depth']:>12.4f}"
        f"{candidate['depth_difference']:>14.4f}"
        f"{str(candidate['is_closer']):>10}"
        f"{candidate['score']:>12.4f}"
    )


print()
print(
    f"Target pixels: "
    f"{roles['target_mask'].sum()}"
)

print(
    f"Occluder pixels: "
    f"{roles['occluder_mask'].sum()}"
)

print(
    f"Context pixels: "
    f"{roles['context_mask'].sum()}"
)