import cv2
import pandas as pd
from pathlib import Path

from src.datasets.kins_loader import KINSLoader
from src.detector.yolo_detector import YOLODetector
from src.detector.target_matcher import match_target
from src.perturbation.perturb import mask_region
from src.perturbation.random_baseline import create_random_local_mask
from src.metrics.detection_metrics import calculate_detection_metrics


# ==========================================================
# Configuration
# ==========================================================

ANNOTATION_PATH = "data/annotations/update_train_2020.json"
IMAGE_DIR = "data/images/training/image_2"

VALIDATED_RESULTS_CSV = "results/batch_results_validated.csv"

MODEL_PATH = "yolo11n.pt"

MATCH_IOU_THRESHOLD = 0.30

NUM_RANDOM_MASKS = 10

RANDOM_MASK_FRACTION = 0.25


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
# Load validated target IDs
# ==========================================================

validated_df = pd.read_csv(
    VALIDATED_RESULTS_CSV
)

target_ids = (
    validated_df["target_ann_id"]
    .drop_duplicates()
    .astype(int)
    .tolist()
)


print("=" * 70)
print("RANDOM MASKING BASELINE")
print("=" * 70)

print(
    f"Validated targets: {len(target_ids)}"
)

print(
    f"Random masks per target: {NUM_RANDOM_MASKS}"
)

print()


results_rows = []


# ==========================================================
# Process targets
# ==========================================================

for target_index, target_id in enumerate(target_ids):

    target_ann = loader.get_annotation_by_id(
        target_id
    )

    if target_ann is None:
        print(
            f"Skipping target {target_id}: "
            f"annotation not found"
        )
        continue


    target_class = loader.get_category_name(
        target_ann["category_id"]
    )


    image_info = loader.get_image(
        target_ann["image_id"]
    )

    if image_info is None:
        print(
            f"Skipping target {target_id}: "
            f"image info not found"
        )
        continue


    file_name = image_info["file_name"]

    image_path = (
        Path(IMAGE_DIR)
        / file_name
    )


    print()
    print("=" * 70)

    print(
        f"TARGET "
        f"{target_index + 1}/{len(target_ids)}"
    )

    print("=" * 70)

    print(
        f"Target ID: {target_id}"
    )

    print(
        f"Image: {file_name}"
    )


    # ======================================================
    # Load image
    # ======================================================

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        print(
            f"Skipping: could not load "
            f"{image_path}"
        )
        continue


    height, width = image.shape[:2]


    # ======================================================
    # Original YOLO detection
    # ======================================================

    original_detections = detector.detect(
        image
    )


    original_match, original_match_iou = (
        match_target(
            target_annotation=target_ann,
            target_class=target_class,
            detections=original_detections,
            min_iou=MATCH_IOU_THRESHOLD
        )
    )


    if original_match is None:

        print(
            f"Skipping: original target "
            f"could not be matched "
            f"(best IoU={original_match_iou:.4f})"
        )

        continue


    print(
        f"Original confidence: "
        f"{original_match['confidence']:.4f}"
    )

    print(
        f"Original match IoU: "
        f"{original_match_iou:.4f}"
    )


    # ======================================================
    # Run random masks
    # ======================================================

    for run_index in range(
        NUM_RANDOM_MASKS
    ):

        random_mask = create_random_local_mask(
            target_annotation=target_ann,
            height=height,
            width=width,
            scale=1.5,
            mask_fraction=RANDOM_MASK_FRACTION
        )


        perturbed_image = mask_region(
            image,
            random_mask
        )


        detections = detector.detect(
            perturbed_image
        )


        matched_detection, match_iou = (
            match_target(
                target_annotation=target_ann,
                target_class=target_class,
                detections=detections,
                min_iou=MATCH_IOU_THRESHOLD
            )
        )


        metrics = calculate_detection_metrics(
            original_detection=original_match,
            perturbed_detection=matched_detection
        )


        record = {
            "image_id": target_ann["image_id"],
            "file_name": file_name,

            "target_ann_id": target_ann["id"],
            "target_class": target_class,

            "baseline_type": "random_local_mask",

            "run_index": run_index + 1,

            "mask_fraction": (
                RANDOM_MASK_FRACTION
            ),

            "original_confidence": (
                metrics[
                    "original_confidence"
                ]
            ),

            "perturbed_confidence": (
                metrics[
                    "perturbed_confidence"
                ]
            ),

            "confidence_change": (
                metrics[
                    "confidence_change"
                ]
            ),

            "bbox_iou": (
                metrics[
                    "bbox_iou"
                ]
            ),

            "survived": (
                metrics[
                    "survived"
                ]
            ),

            "original_match_iou": (
                original_match_iou
            ),

            "perturbed_match_iou": (
                match_iou
            )
        }


        results_rows.append(
            record
        )


        if metrics["survived"]:

            print(
                f"Run {run_index + 1:02d} | "
                f"survived=True | "
                f"Δconf="
                f"{metrics['confidence_change']:.4f} | "
                f"bbox IoU="
                f"{metrics['bbox_iou']:.4f}"
            )

        else:

            print(
                f"Run {run_index + 1:02d} | "
                f"survived=False | "
                f"best match IoU="
                f"{match_iou:.4f}"
            )


# ==========================================================
# Save baseline results
# ==========================================================

Path("results").mkdir(
    exist_ok=True
)


baseline_df = pd.DataFrame(
    results_rows
)


output_csv = (
    "results/random_baseline_results.csv"
)


baseline_df.to_csv(
    output_csv,
    index=False
)


print()
print("=" * 70)
print("BASELINE COMPLETE")
print("=" * 70)

print(
    f"Total baseline rows: "
    f"{len(baseline_df)}"
)

unique_targets = (
    baseline_df["target_ann_id"].nunique()
    if not baseline_df.empty
    else 0
)

print(
    f"Unique targets processed: {unique_targets}"
)

print(
    f"Saved to: {output_csv}"
)