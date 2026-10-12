import cv2
import pandas as pd
from pathlib import Path

from src.datasets.kins_loader import KINSLoader
from src.detector.yolo_detector import YOLODetector
from src.detector.target_matcher import match_target
from src.perturbation.perturb import mask_region
from src.perturbation.synthetic_occlusion import (
    create_synthetic_occlusion_mask
)
from src.metrics.detection_metrics import (
    calculate_detection_metrics
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

VALIDATED_RESULTS_CSV = (
    "results/batch_results_validated.csv"
)

MODEL_PATH = "yolo11n.pt"

MATCH_IOU_THRESHOLD = 0.30


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

detector = YOLODetector(
    MODEL_PATH
)


# ==========================================================
# Load validated targets
# ==========================================================

validated_df = pd.read_csv(
    VALIDATED_RESULTS_CSV
)

target_ids = (
    validated_df[
        "target_ann_id"
    ]
    .drop_duplicates()
    .astype(int)
    .tolist()
)


print("=" * 70)
print("CONTROLLED SYNTHETIC OCCLUSION")
print("=" * 70)

print(
    f"Validated targets: "
    f"{len(target_ids)}"
)

print(
    f"Occlusion levels: "
    f"{OCCLUSION_LEVELS}"
)

print()


results_rows = []


# ==========================================================
# Process targets
# ==========================================================

for target_index, target_id in enumerate(
    target_ids
):

    target_ann = (
        loader.get_annotation_by_id(
            target_id
        )
    )

    if target_ann is None:
        continue


    target_class = (
        loader.get_category_name(
            target_ann["category_id"]
        )
    )


    image_info = loader.get_image(
        target_ann["image_id"]
    )

    if image_info is None:
        continue


    file_name = (
        image_info["file_name"]
    )

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


    print()
    print("=" * 70)

    print(
        f"TARGET "
        f"{target_index + 1}/"
        f"{len(target_ids)}"
    )

    print("=" * 70)

    print(
        f"Target ID: {target_id}"
    )

    print(
        f"Image: {file_name}"
    )


    # ======================================================
    # Original detection
    # ======================================================

    original_detections = (
        detector.detect(
            image
        )
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
            "Skipping: original target "
            "could not be matched"
        )

        continue


    print(
        f"Original confidence: "
        f"{original_match['confidence']:.4f}"
    )

    print(
        f"Original match IoU: "
        f"{original_iou:.4f}"
    )


    # ======================================================
    # Synthetic occlusion levels
    # ======================================================

    for occlusion_fraction in (
        OCCLUSION_LEVELS
    ):

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


        detections = detector.detect(
            occluded_image
        )


        matched_detection, match_iou = (
            match_target(
                target_annotation=target_ann,
                target_class=target_class,
                detections=detections,
                min_iou=MATCH_IOU_THRESHOLD
            )
        )


        metrics = (
            calculate_detection_metrics(
                original_detection=(
                    original_match
                ),
                perturbed_detection=(
                    matched_detection
                )
            )
        )


        record = {
            "image_id": (
                target_ann["image_id"]
            ),

            "file_name": file_name,

            "target_ann_id": (
                target_ann["id"]
            ),

            "target_class": (
                target_class
            ),

            "synthetic_occlusion_fraction": (
                occlusion_fraction
            ),

            "synthetic_occlusion_percent": (
                occlusion_fraction
                * 100
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
                original_iou
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
                f"Occlusion "
                f"{int(occlusion_fraction * 100):02d}%"
                f" | survived=True"
                f" | Δconf="
                f"{metrics['confidence_change']:.4f}"
                f" | bbox IoU="
                f"{metrics['bbox_iou']:.4f}"
            )

        else:

            print(
                f"Occlusion "
                f"{int(occlusion_fraction * 100):02d}%"
                f" | survived=False"
                f" | best match IoU="
                f"{match_iou:.4f}"
            )


# ==========================================================
# Save results
# ==========================================================

Path("results").mkdir(
    exist_ok=True
)


results_df = pd.DataFrame(
    results_rows
)


output_csv = (
    "results/"
    "synthetic_occlusion_results.csv"
)


results_df.to_csv(
    output_csv,
    index=False
)


print()
print("=" * 70)
print("SYNTHETIC OCCLUSION COMPLETE")
print("=" * 70)

print(
    f"Total rows: "
    f"{len(results_df)}"
)

unique_targets = (
    results_df[
        "target_ann_id"
    ].nunique()
    if not results_df.empty
    else 0
)

print(
    f"Unique targets processed: "
    f"{unique_targets}"
)

print(
    f"Saved to: "
    f"{output_csv}"
)