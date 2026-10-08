import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.datasets.kins_loader import KINSLoader
from src.detector.yolo_detector import YOLODetector
from src.detector.target_matcher import match_target
from src.roles.scene_roles import build_scene_roles


# ==========================================================
# Configuration
# ==========================================================

ANNOTATION_PATH = "data/annotations/update_train_2020.json"
IMAGE_DIR = "data/images/training/image_2"
BATCH_RESULTS_PATH = "results/batch_results.csv"

OUTPUT_DIR = Path("results/batch_validation")

MODEL_PATH = "yolo11n.pt"
MATCH_IOU_THRESHOLD = 0.30


# ==========================================================
# Setup
# ==========================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

loader = KINSLoader(
    ANNOTATION_PATH
)

detector = YOLODetector(
    MODEL_PATH
)


# ==========================================================
# Read processed target IDs
# ==========================================================

df = pd.read_csv(
    BATCH_RESULTS_PATH
)

target_ids = (
    df["target_ann_id"]
    .drop_duplicates()
    .tolist()
)

print(
    f"Targets to validate: {len(target_ids)}"
)


# ==========================================================
# Helper: overlay mask
# ==========================================================

def create_mask_overlay(
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
        0.5 * overlay[mask].astype(np.float32)
        + 0.5 * color
    ).astype(np.uint8)

    return overlay


# ==========================================================
# Process targets
# ==========================================================

for index, target_id in enumerate(target_ids):

    target_ann = loader.get_annotation_by_id(
        int(target_id)
    )

    if target_ann is None:
        continue


    target_class = loader.get_category_name(
        target_ann["category_id"]
    )


    image_info = loader.get_image(
        target_ann["image_id"]
    )

    file_name = image_info["file_name"]

    image_path = Path(
        IMAGE_DIR
    ) / file_name


    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        print(
            f"Could not load {image_path}"
        )
        continue


    height, width = image.shape[:2]

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    # ======================================================
    # Build scene roles
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


    occluder_ann = roles[
        "occluder_annotation"
    ]


    # ======================================================
    # YOLO detection
    # ======================================================

    detections = detector.detect(
        image
    )


    matched_detection, match_iou = match_target(
        target_annotation=target_ann,
        target_class=target_class,
        detections=detections,
        min_iou=MATCH_IOU_THRESHOLD
    )


    # ======================================================
    # Original image with bounding boxes
    # ======================================================

    bbox_image = image_rgb.copy()


    # KINS visible bbox
    x, y, w, h = target_ann["i_bbox"]

    cv2.rectangle(
        bbox_image,
        (int(x), int(y)),
        (int(x + w), int(y + h)),
        (0, 255, 0),
        2
    )


    # YOLO bbox
    if matched_detection is not None:

        x1, y1, x2, y2 = (
            matched_detection["bbox"]
        )

        cv2.rectangle(
            bbox_image,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (255, 0, 0),
            2
        )


    # ======================================================
    # Mask overlays
    # ======================================================

    target_overlay = create_mask_overlay(
        image_rgb,
        roles["target_mask"],
        color=[255, 0, 0]
    )


    occluder_overlay = create_mask_overlay(
        image_rgb,
        roles["occluder_mask"],
        color=[255, 255, 0]
    )


    context_overlay = create_mask_overlay(
        image_rgb,
        roles["context_mask"],
        color=[0, 0, 255]
    )


    # ======================================================
    # Occluder information
    # ======================================================

    if occluder_ann is not None:

        occluder_id = occluder_ann["id"]

        occluder_class = (
            loader.get_category_name(
                occluder_ann["category_id"]
            )
        )

    else:

        occluder_id = None
        occluder_class = None


    # ======================================================
    # Plot
    # ======================================================

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 8)
    )


    # Original + boxes
    axes[0, 0].imshow(
        bbox_image
    )

    axes[0, 0].set_title(
        f"Original\n"
        f"Target {target_id} ({target_class}) | "
        f"YOLO IoU={match_iou:.3f}\n"
        f"Green=KINS | Blue=YOLO"
    )

    axes[0, 0].axis("off")


    # Target
    axes[0, 1].imshow(
        target_overlay
    )

    axes[0, 1].set_title(
        "Target Mask (Red)"
    )

    axes[0, 1].axis("off")


    # Occluder
    axes[1, 0].imshow(
        occluder_overlay
    )

    axes[1, 0].set_title(
        f"Occluder (Yellow)\n"
        f"Ann {occluder_id} | "
        f"{occluder_class}"
    )

    axes[1, 0].axis("off")


    # Context
    axes[1, 1].imshow(
        context_overlay
    )

    axes[1, 1].set_title(
        "Local Context (Blue)"
    )

    axes[1, 1].axis("off")


    plt.tight_layout()


    output_path = (
        OUTPUT_DIR
        / f"target_{target_id}.png"
    )


    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


    print(
        f"[{index + 1}/{len(target_ids)}] "
        f"Saved {output_path}"
    )


print()
print("=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)

print(
    f"Images saved to: {OUTPUT_DIR}"
)