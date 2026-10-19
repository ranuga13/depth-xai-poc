import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from src.datasets.kins_loader import KINSLoader
from src.perturbation.perturb import mask_region
from src.perturbation.synthetic_occlusion import (
    create_synthetic_occlusion_mask
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

OUTPUT_DIR = Path(
    "results/synthetic_occlusion_visualization"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Choose one or two target IDs here
TARGET_IDS = [
    68,
    148
]


OCCLUSION_LEVELS = [
    0.10,
    0.30,
    0.50,
    0.70
]


# ==========================================================
# Setup
# ==========================================================

loader = KINSLoader(
    ANNOTATION_PATH
)


# ==========================================================
# Process targets
# ==========================================================

for target_id in TARGET_IDS:

    target_ann = loader.get_annotation_by_id(
        target_id
    )

    if target_ann is None:
        print(
            f"Target {target_id} not found"
        )
        continue


    image_info = loader.get_image(
        target_ann["image_id"]
    )

    if image_info is None:
        print(
            f"Image info not found for target {target_id}"
        )
        continue


    file_name = image_info["file_name"]

    image_path = (
        Path(IMAGE_DIR)
        / file_name
    )


    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        print(
            f"Could not load image: {image_path}"
        )
        continue


    height, width = image.shape[:2]


    # Convert original image to RGB
    original_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    images_to_show = [
        original_rgb
    ]

    titles = [
        "Original"
    ]


    # ======================================================
    # Generate occluded versions
    # ======================================================

    for occlusion_fraction in OCCLUSION_LEVELS:

        synthetic_mask = (
            create_synthetic_occlusion_mask(
                target_annotation=target_ann,
                height=height,
                width=width,
                occlusion_fraction=(
                    occlusion_fraction
                )
            )
        )


        occluded_image = mask_region(
            image,
            synthetic_mask
        )


        occluded_rgb = cv2.cvtColor(
            occluded_image,
            cv2.COLOR_BGR2RGB
        )


        images_to_show.append(
            occluded_rgb
        )

        titles.append(
            f"{int(occlusion_fraction * 100)}% Occlusion"
        )


    # ======================================================
    # Plot
    # ======================================================

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(20, 5)
    )


    for i in range(5):

        axes[i].imshow(
            images_to_show[i]
        )

        axes[i].set_title(
            titles[i]
        )

        axes[i].axis(
            "off"
        )


    plt.suptitle(
        f"Target {target_id} | {file_name}",
        fontsize=14
    )

    plt.tight_layout()


    output_path = (
        OUTPUT_DIR
        / f"target_{target_id}_occlusion_levels.png"
    )


    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


    print(
        f"Saved: {output_path}"
    )


print()
print(
    f"Visualizations saved to: {OUTPUT_DIR}"
)