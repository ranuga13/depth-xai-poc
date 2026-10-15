import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from src.datasets.kins_loader import KINSLoader
from src.roles.scene_roles import build_scene_roles


# ==========================================================
# Configuration
# ==========================================================

ANNOTATION_PATH = "data/annotations/update_train_2020.json"
IMAGE_DIR = "data/images/training/image_2"

DRISE_DIR = Path(
    "results/drise_batch"
)

RANK_PATH = (
    "results/drise_rank_analysis/"
    "drise_rank_correlation.csv"
)

OUTPUT_DIR = Path(
    "results/drise_visual_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TARGET_IDS = [
    68,
    148,
    743,
    1470,
    1873
]


# ==========================================================
# Setup
# ==========================================================

loader = KINSLoader(
    ANNOTATION_PATH
)

rank_df = pd.read_csv(
    RANK_PATH
)


# ==========================================================
# Helper
# ==========================================================

def overlay_mask(
    image_rgb,
    mask,
    color
):
    overlay = image_rgb.copy()

    color = np.array(
        color,
        dtype=np.float32
    )

    overlay[mask] = (
        0.5
        * overlay[mask].astype(
            np.float32
        )
        + 0.5
        * color
    ).astype(
        np.uint8
    )

    return overlay


# ==========================================================
# Process targets
# ==========================================================

for target_id in TARGET_IDS:

    target_ann = loader.get_annotation_by_id(
        target_id
    )

    if target_ann is None:
        print(
            f"Skipping {target_id}: annotation not found"
        )
        continue


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
        print(
            f"Skipping {target_id}: image not found"
        )
        continue


    height, width = image.shape[:2]

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    # ======================================================
    # Scene roles
    # ======================================================

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


    target_overlay = overlay_mask(
        image_rgb,
        roles["target_mask"],
        [255, 0, 0]
    )

    occluder_overlay = overlay_mask(
        image_rgb,
        roles["occluder_mask"],
        [255, 255, 0]
    )

    context_overlay = overlay_mask(
        image_rgb,
        roles["context_mask"],
        [0, 0, 255]
    )


    # ======================================================
    # D-RISE
    # ======================================================

    saliency_path = (
        DRISE_DIR
        / f"target_{target_id}_drise.npy"
    )

    if not saliency_path.exists():
        print(
            f"Skipping {target_id}: D-RISE map missing"
        )
        continue


    saliency = np.load(
        saliency_path
    )


    # ======================================================
    # Rankings
    # ======================================================

    rank_row = rank_df[
        rank_df[
            "target_ann_id"
        ] == target_id
    ]


    if not rank_row.empty:

        rank_row = rank_row.iloc[0]

        role_ranking = (
            rank_row[
                "role_ranking"
            ]
        )

        drise_ranking = (
            rank_row[
                "drise_ranking"
            ]
        )

        correlation = (
            rank_row[
                "spearman_correlation"
            ]
        )

    else:

        role_ranking = "N/A"
        drise_ranking = "N/A"
        correlation = np.nan


    # ======================================================
    # Plot
    # ==========================================================

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(22, 5)
    )


    axes[0].imshow(
        image_rgb
    )

    axes[0].set_title(
        "Original"
    )


    axes[1].imshow(
        target_overlay
    )

    axes[1].set_title(
        "Target"
    )


    axes[2].imshow(
        occluder_overlay
    )

    axes[2].set_title(
        "Occluder"
    )


    axes[3].imshow(
        context_overlay
    )

    axes[3].set_title(
        "Context"
    )


    axes[4].imshow(
        image_rgb
    )

    axes[4].imshow(
        saliency,
        cmap="jet",
        alpha=0.5
    )

    axes[4].set_title(
        "D-RISE"
    )


    for ax in axes:
        ax.axis(
            "off"
        )


    fig.suptitle(
        (
            f"Target {target_id}\n"
            f"Role-aware: {role_ranking} | "
            f"D-RISE: {drise_ranking} | "
            f"Spearman: {correlation:.2f}"
        ),
        fontsize=13
    )


    plt.tight_layout()


    output_path = (
        OUTPUT_DIR
        / f"target_{target_id}_comparison.png"
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
    f"All visual comparisons saved to: "
    f"{OUTPUT_DIR}"
)