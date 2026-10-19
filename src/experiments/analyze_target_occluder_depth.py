import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.datasets.kins_loader import KINSLoader
from src.roles.scene_roles import build_scene_roles


# ==========================================================
# Configuration
# ==========================================================

ANNOTATION_PATH = "data/annotations/update_train_2020.json"
IMAGE_DIR = "data/images/training/image_2"

TARGET_ID = 68

DEPTH_PATH = (
    "results/depth_single/"
    "target_68_depth.npy"
)

OUTPUT_DIR = Path(
    "results/depth_role_analysis"
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
# Load depth map
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
# Build target / occluder / context masks
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
# Depth statistics
# ==========================================================

def get_depth_stats(
    name,
    mask,
    depth_map
):

    values = depth_map[
        mask
    ]

    if len(values) == 0:

        return {
            "role": name,
            "mean": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
            "pixels": 0
        }


    return {
        "role": name,

        "mean": float(
            values.mean()
        ),

        "median": float(
            np.median(
                values
            )
        ),

        "min": float(
            values.min()
        ),

        "max": float(
            values.max()
        ),

        "pixels": int(
            len(values)
        )
    }


target_stats = get_depth_stats(
    "target",
    target_mask,
    depth_map
)

occluder_stats = get_depth_stats(
    "occluder",
    occluder_mask,
    depth_map
)

context_stats = get_depth_stats(
    "context",
    context_mask,
    depth_map
)


# ==========================================================
# Print results
# ==========================================================

print()
print("=" * 80)
print("TARGET / OCCLUDER DEPTH ANALYSIS")
print("=" * 80)

print(
    f"Target ID: {TARGET_ID}"
)

print(
    f"Image: {file_name}"
)

print()

print(
    f"{'Role':<12}"
    f"{'Pixels':>10}"
    f"{'Mean':>12}"
    f"{'Median':>12}"
    f"{'Min':>12}"
    f"{'Max':>12}"
)

print(
    "-" * 70
)


for stats in [
    target_stats,
    occluder_stats,
    context_stats
]:

    print(
        f"{stats['role']:<12}"
        f"{stats['pixels']:>10}"
        f"{stats['mean']:>12.4f}"
        f"{stats['median']:>12.4f}"
        f"{stats['min']:>12.4f}"
        f"{stats['max']:>12.4f}"
    )


# ==========================================================
# Compare target vs occluder
# ==========================================================

target_median = (
    target_stats[
        "median"
    ]
)

occluder_median = (
    occluder_stats[
        "median"
    ]
)


print()
print("=" * 80)
print("DEPTH COMPARISON")
print("=" * 80)

print(
    f"Target median depth: "
    f"{target_median:.4f}"
)

print(
    f"Occluder median depth: "
    f"{occluder_median:.4f}"
)

print(
    f"Difference "
    f"(occluder - target): "
    f"{occluder_median - target_median:.4f}"
)


# ==========================================================
# Visualization
# ==========================================================

depth_vis = (
    depth_map
    - depth_map.min()
)

depth_range = (
    depth_map.max()
    - depth_map.min()
)

if depth_range > 0:

    depth_vis = (
        depth_vis
        / depth_range
    )


fig, axes = plt.subplots(
    1,
    4,
    figsize=(20, 5)
)


axes[0].imshow(
    depth_vis,
    cmap="inferno"
)

axes[0].set_title(
    "Depth Map"
)


axes[1].imshow(
    depth_vis,
    cmap="inferno"
)

axes[1].imshow(
    target_mask,
    cmap="Reds",
    alpha=0.45
)

axes[1].set_title(
    f"Target\nmedian={target_median:.3f}"
)


axes[2].imshow(
    depth_vis,
    cmap="inferno"
)

axes[2].imshow(
    occluder_mask,
    cmap="Blues",
    alpha=0.45
)

axes[2].set_title(
    f"Occluder\nmedian={occluder_median:.3f}"
)


axes[3].imshow(
    depth_vis,
    cmap="inferno"
)

axes[3].imshow(
    context_mask,
    cmap="Greens",
    alpha=0.35
)

axes[3].set_title(
    f"Context\nmedian={context_stats['median']:.3f}"
)


for ax in axes:

    ax.axis(
        "off"
    )


plt.suptitle(
    f"Depth Role Analysis | Target {TARGET_ID}"
)

plt.tight_layout()


output_path = (
    OUTPUT_DIR
    / f"target_{TARGET_ID}_depth_roles.png"
)


plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print()
print(
    f"Saved visualization: "
    f"{output_path}"
)