import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from src.datasets.kins_loader import KINSLoader
from src.depth.depth_estimator import DepthEstimator


ANNOTATION_PATH = (
    "data/annotations/update_train_2020.json"
)

IMAGE_DIR = (
    "data/images/training/image_2"
)

TARGET_ID = 68

OUTPUT_DIR = Path(
    "results/depth_single"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


loader = KINSLoader(
    ANNOTATION_PATH
)

depth_estimator = DepthEstimator()


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


print("=" * 70)
print("DEPTH ANYTHING V2 TEST")
print("=" * 70)

print(
    f"Target ID: {TARGET_ID}"
)

print(
    f"Image: {file_name}"
)


depth_map = depth_estimator.predict(
    image
)


print(
    f"Depth shape: {depth_map.shape}"
)

print(
    f"Depth min: {depth_map.min():.4f}"
)

print(
    f"Depth max: {depth_map.max():.4f}"
)

print(
    f"Depth median: {np.median(depth_map):.4f}"
)


# ==========================================================
# Resize if necessary
# ==========================================================

height, width = image.shape[:2]

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
# Normalize only for visualization
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


image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# ==========================================================
# Plot
# ==========================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 6)
)


axes[0].imshow(
    image_rgb
)

axes[0].set_title(
    "Original Image"
)

axes[0].axis(
    "off"
)


depth_plot = axes[1].imshow(
    depth_vis,
    cmap="inferno"
)

axes[1].set_title(
    "Relative Depth Map"
)

axes[1].axis(
    "off"
)

fig.colorbar(
    depth_plot,
    ax=axes[1],
    fraction=0.046,
    pad=0.04
)


axes[2].imshow(
    image_rgb
)

axes[2].imshow(
    depth_vis,
    cmap="inferno",
    alpha=0.5
)

axes[2].set_title(
    "Depth Overlay"
)

axes[2].axis(
    "off"
)


plt.suptitle(
    f"Target {TARGET_ID} | {file_name}"
)

plt.tight_layout()


output_path = (
    OUTPUT_DIR
    / f"target_{TARGET_ID}_depth.png"
)

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


np.save(
    OUTPUT_DIR
    / f"target_{TARGET_ID}_depth.npy",
    depth_map
)


print()
print("=" * 70)
print("DEPTH TEST COMPLETE")
print("=" * 70)

print(
    f"Saved visualization: {output_path}"
)