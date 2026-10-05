import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pycocotools import mask as mask_utils

from datasets.kins_loader import KINSLoader
from datasets.sample_filter import find_suitable_targets


ANNOTATION_PATH = "data/annotations/update_train_2020.json"
IMAGE_DIR = Path("data/images/training/image_2")

OUTPUT_DIR = Path("results/filtered_samples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def polygon_to_mask(segmentation, height, width):
    rles = mask_utils.frPyObjects(
        segmentation,
        height,
        width
    )

    rle = mask_utils.merge(rles)

    mask = mask_utils.decode(rle)

    return mask.astype(bool)


loader = KINSLoader(
    ANNOTATION_PATH
)


targets = find_suitable_targets(
    loader=loader,
    target_class="car",
    min_occlusion=0.20,
    max_occlusion=0.60,
    min_width=100,
    min_height=60,
    min_area=6000,
    max_targets=10
)


for item in targets:

    ann = item["annotation"]

    image_info = loader.get_image(
        ann["image_id"]
    )

    image_path = (
        IMAGE_DIR
        / image_info["file_name"]
    )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        print(f"Could not load {image_path}")
        continue

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    height, width = image.shape[:2]


    # -------------------------
    # Masks
    # -------------------------

    amodal_mask = polygon_to_mask(
        ann["a_segm"],
        height,
        width
    )

    visible_mask = polygon_to_mask(
        ann["i_segm"],
        height,
        width
    )


    # -------------------------
    # Overlay
    # -------------------------

    overlay = image_rgb.copy()

    # Amodal = red
    overlay[amodal_mask] = (
        0.6 * overlay[amodal_mask]
        + 0.4 * np.array([255, 0, 0])
    ).astype(np.uint8)

    # Visible = green
    overlay[visible_mask] = (
        0.5 * overlay[visible_mask]
        + 0.5 * np.array([0, 255, 0])
    ).astype(np.uint8)


    # -------------------------
    # Plot
    # -------------------------

    fig = plt.figure(
        figsize=(16, 8)
    )

    plt.subplot(1, 4, 1)
    plt.imshow(image_rgb)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(amodal_mask, cmap="gray")
    plt.title("Amodal Mask")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(visible_mask, cmap="gray")
    plt.title("Visible Mask")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(overlay)
    plt.title(
        "Red = Amodal | Green = Visible"
    )
    plt.axis("off")


    fig.suptitle(
        (
            f"{image_info['file_name']} | "
            f"Ann ID: {ann['id']} | "
            f"Occlusion: "
            f"{item['occlusion_ratio']:.2f}"
        )
    )

    plt.tight_layout()


    output_path = (
        OUTPUT_DIR
        / f"{image_info['file_name']}_ann_{ann['id']}.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )