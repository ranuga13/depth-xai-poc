import numpy as np
import cv2
from pathlib import Path

from src.datasets.kins_loader import KINSLoader
from src.roles.scene_roles import build_scene_roles


# ==========================================================
# Configuration
# ==========================================================

ANNOTATION_PATH = "data/annotations/update_train_2020.json"
IMAGE_DIR = "data/images/training/image_2"

TARGET_ANN_ID = 68

DRISE_DIR = Path(
    "results/drise"
)


# ==========================================================
# Setup
# ==========================================================

loader = KINSLoader(
    ANNOTATION_PATH
)


target_ann = loader.get_annotation_by_id(
    TARGET_ANN_ID
)

if target_ann is None:
    raise ValueError(
        f"Target {TARGET_ANN_ID} not found"
    )


image_info = loader.get_image(
    target_ann["image_id"]
)

file_name = image_info["file_name"]

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
# Load D-RISE saliency map
# ==========================================================

saliency_path = (
    DRISE_DIR
    / f"target_{TARGET_ANN_ID}_drise.npy"
)


saliency = np.load(
    saliency_path
)


print("=" * 70)
print("D-RISE ROLE ANALYSIS")
print("=" * 70)

print(
    f"Target ID: {TARGET_ANN_ID}"
)

print(
    f"Image: {file_name}"
)


# ==========================================================
# Build role masks
# ==========================================================

image_annotations = (
    loader.get_annotations_for_image(
        target_ann["image_id"]
    )
)


roles = build_scene_roles(
    target_annotation=target_ann,
    image_annotations=image_annotations,
    height=height,
    width=width
)


target_mask = roles[
    "target_mask"
]

occluder_mask = roles[
    "occluder_mask"
]

context_mask = roles[
    "context_mask"
]


# ==========================================================
# Helper
# ==========================================================

def analyze_region(
    name,
    mask,
    saliency
):
    pixel_count = int(
        mask.sum()
    )

    if pixel_count == 0:
        return {
            "role": name,
            "pixel_count": 0,
            "mean_saliency": 0.0,
            "total_saliency": 0.0
        }


    values = saliency[
        mask
    ]


    return {
        "role": name,

        "pixel_count": (
            pixel_count
        ),

        "mean_saliency": float(
            values.mean()
        ),

        "total_saliency": float(
            values.sum()
        )
    }


# ==========================================================
# Analyze roles
# ==========================================================

results = []


results.append(
    analyze_region(
        "target",
        target_mask,
        saliency
    )
)


results.append(
    analyze_region(
        "occluder",
        occluder_mask,
        saliency
    )
)


results.append(
    analyze_region(
        "context",
        context_mask,
        saliency
    )
)


# ==========================================================
# Total role saliency
# ==========================================================

role_total_saliency = sum(
    item[
        "total_saliency"
    ]
    for item in results
)


for item in results:

    if role_total_saliency > 0:

        item[
            "role_saliency_percent"
        ] = (
            item[
                "total_saliency"
            ]
            / role_total_saliency
            * 100
        )

    else:

        item[
            "role_saliency_percent"
        ] = 0.0


# ==========================================================
# Print
# ==========================================================

print()

print(
    f"{'Role':<12}"
    f"{'Pixels':>12}"
    f"{'Mean Saliency':>18}"
    f"{'Total Saliency':>18}"
    f"{'Role %':>12}"
)

print(
    "-" * 72
)


for item in results:

    print(
        f"{item['role']:<12}"
        f"{item['pixel_count']:>12}"
        f"{item['mean_saliency']:>18.4f}"
        f"{item['total_saliency']:>18.2f}"
        f"{item['role_saliency_percent']:>11.2f}%"
    )


# ==========================================================
# Find highest mean-saliency role
# ==========================================================

best_role = max(
    results,
    key=lambda item:
    item[
        "mean_saliency"
    ]
)


print()
print(
    f"Highest mean-saliency role: "
    f"{best_role['role']}"
)

print(
    f"Mean saliency: "
    f"{best_role['mean_saliency']:.4f}"
)