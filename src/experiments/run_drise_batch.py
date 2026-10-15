import cv2
import numpy as np
from pathlib import Path
import pandas as pd

from src.datasets.kins_loader import KINSLoader
from src.detector.yolo_detector import YOLODetector
from src.detector.target_matcher import (
    match_target,
    calculate_iou
)


# ==========================================================
# Configuration
# ==========================================================

ANNOTATION_PATH = "data/annotations/update_train_2020.json"
IMAGE_DIR = "data/images/training/image_2"

MODEL_PATH = "yolo11n.pt"

VALIDATED_RESULTS_PATH = (
    "results/batch_results_validated.csv"
)

validated_df = pd.read_csv(
    VALIDATED_RESULTS_PATH
)

TARGET_IDS = sorted(
    validated_df[
        "target_ann_id"
    ].unique().tolist()
)

print(
    f"Validated targets: {len(TARGET_IDS)}"
)

RANDOM_SEED = 42

np.random.seed(
    RANDOM_SEED
)

MATCH_IOU_THRESHOLD = 0.30

NUM_MASKS = 500
GRID_SIZE = 8
MASK_PROBABILITY = 0.5

OUTPUT_DIR = Path(
    "results/drise_full"
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

detector = YOLODetector(
    MODEL_PATH
)


# ==========================================================
# Helpers
# ==========================================================

def generate_random_mask(
    height,
    width,
    grid_size=8,
    probability=0.5
):
    small_mask = (
        np.random.rand(
            grid_size,
            grid_size
        )
        < probability
    ).astype(np.float32)

    return cv2.resize(
        small_mask,
        (width, height),
        interpolation=cv2.INTER_LINEAR
    )


def calculate_drise_similarity(
    original_detection,
    detections
):
    best_score = 0.0

    original_bbox = (
        original_detection["bbox"]
    )

    original_class = (
        original_detection["class_name"]
    )

    for detection in detections:

        if (
            detection["class_name"]
            != original_class
        ):
            continue

        iou = calculate_iou(
            original_bbox,
            detection["bbox"]
        )

        score = (
            iou
            * detection["confidence"]
        )

        if score > best_score:
            best_score = score

    return best_score


# ==========================================================
# Run D-RISE
# ==========================================================

for target_index, target_id in enumerate(
    TARGET_IDS
):

    target_ann = loader.get_annotation_by_id(
        target_id
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

    image_path = (
        Path(IMAGE_DIR)
        / file_name
    )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        continue

    height, width = image.shape[:2]

    original_detections = detector.detect(
        image
    )

    original_match, original_iou = (
        match_target(
            target_annotation=target_ann,
            target_class=target_class,
            detections=original_detections,
            min_iou=MATCH_IOU_THRESHOLD
        )
    )

    if original_match is None:

        print(
            f"Skipping {target_id}: "
            f"original target not matched"
        )

        continue


    print()
    print("=" * 70)

    print(
        f"D-RISE TARGET "
        f"{target_index + 1}/"
        f"{len(TARGET_IDS)}"
    )

    print("=" * 70)

    print(
        f"Target ID: {target_id}"
    )

    print(
        f"Original confidence: "
        f"{original_match['confidence']:.4f}"
    )


    saliency = np.zeros(
        (height, width),
        dtype=np.float32
    )


    for mask_index in range(
        NUM_MASKS
    ):

        random_mask = generate_random_mask(
            height=height,
            width=width,
            grid_size=GRID_SIZE,
            probability=MASK_PROBABILITY
        )

        masked_image = (
            image.astype(np.float32)
            * random_mask[..., None]
        ).astype(np.uint8)

        detections = detector.detect(
            masked_image
        )

        similarity = (
            calculate_drise_similarity(
                original_detection=original_match,
                detections=detections
            )
        )

        saliency += (
            similarity
            * random_mask
        )

        if (
            (mask_index + 1) % 100
            == 0
        ):

            print(
                f"Processed "
                f"{mask_index + 1}/"
                f"{NUM_MASKS}"
            )


    # ======================================================
    # Normalize
    # ======================================================

    saliency /= NUM_MASKS

    minimum = saliency.min()
    maximum = saliency.max()

    if maximum > minimum:

        saliency = (
            saliency - minimum
        ) / (
            maximum - minimum
        )


    # ======================================================
    # Save
    # ======================================================

    np.save(
        OUTPUT_DIR
        / f"target_{target_id}_drise.npy",
        saliency
    )


print()
print("=" * 70)
print("D-RISE BATCH COMPLETE")
print("=" * 70)