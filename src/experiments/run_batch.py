import cv2
import pandas as pd
from pathlib import Path

from src.datasets.kins_loader import KINSLoader
from src.datasets.sample_filter import find_suitable_targets
from src.detector.yolo_detector import YOLODetector
from src.detector.target_matcher import match_target
from src.roles.scene_roles import build_scene_roles
from src.perturbation.perturb import mask_region, blur_region
from src.metrics.detection_metrics import calculate_detection_metrics
from src.experiments.experiment_utils import build_result_record


# ==========================================================
# Configuration
# ==========================================================

ANNOTATION_PATH = "data/annotations/update_train_2020.json"
IMAGE_DIR = "data/images/training/image_2"

MODEL_PATH = "yolo11n.pt"

NUM_TARGETS = 50
MATCH_IOU_THRESHOLD = 0.30


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
# Find suitable targets
# ==========================================================

targets = find_suitable_targets(
    loader=loader,
    target_class="car",
    min_occlusion=0.20,
    max_occlusion=0.60,
    min_width=100,
    min_height=60,
    min_area=6000,
    max_targets=NUM_TARGETS
)

print(
    f"Found {len(targets)} suitable targets"
)


results_rows = []


# ==========================================================
# Process each target
# ==========================================================

for index, target_item in enumerate(targets):

    target_ann = target_item["annotation"]

    target_class = target_item["class_name"]

    occlusion_ratio = target_item["occlusion_ratio"]

    image_info = loader.get_image(
        target_ann["image_id"]
    )

    file_name = image_info["file_name"]

    image_path = str(
        Path(IMAGE_DIR) / file_name
    )

    print()
    print("=" * 70)
    print(
        f"TARGET {index + 1}/{len(targets)}"
    )
    print("=" * 70)

    print(
        f"Annotation ID: {target_ann['id']}"
    )

    print(
        f"Image: {file_name}"
    )

    print(
        f"Occlusion ratio: {occlusion_ratio:.3f}"
    )


    # ======================================================
    # Load image
    # ======================================================

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            f"Skipping: could not load {image_path}"
        )

        continue


    height, width = image.shape[:2]


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


    occluder_ann = roles[
        "occluder_annotation"
    ]


    if occluder_ann is None:

        print(
            "Skipping: no foreground occluder found"
        )

        continue


    occluder_ann_id = occluder_ann["id"]

    occluder_class = loader.get_category_name(
        occluder_ann["category_id"]
    )


    print(
        f"Occluder ID: {occluder_ann_id}"
    )

    print(
        f"Occluder class: {occluder_class}"
    )


    # ======================================================
    # Original detection
    # ======================================================

    original_detections = detector.detect(
        image
    )

    original_match, original_iou = match_target(
        target_annotation=target_ann,
        target_class=target_class,
        detections=original_detections,
        min_iou=MATCH_IOU_THRESHOLD
    )


    if original_match is None:

        print(
            f"Skipping: original target not matched "
            f"(best IoU = {original_iou:.3f})"
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
    # Helper for one perturbation
    # ======================================================

    def run_perturbation(
        role_name,
        perturbation_name,
        perturbed_image
    ):

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


        record = build_result_record(
            image_id=target_ann["image_id"],
            file_name=file_name,
            target_ann_id=target_ann["id"],
            target_class=target_class,
            occluder_ann_id=occluder_ann_id,
            occluder_class=occluder_class,
            occlusion_ratio=occlusion_ratio,
            role=role_name,
            perturbation_type=perturbation_name,
            original_match_iou=original_iou,
            metrics=metrics
        )


        results_rows.append(
            record
        )


        if metrics["survived"]:

            print(
                f"{role_name:10} | "
                f"{perturbation_name:5} | "
                f"survived=True | "
                f"Δconf={metrics['confidence_change']:.4f} | "
                f"bbox IoU={metrics['bbox_iou']:.4f}"
            )

        else:

            print(
                f"{role_name:10} | "
                f"{perturbation_name:5} | "
                f"survived=False | "
                f"best match IoU={match_iou:.4f}"
            )


    # ======================================================
    # Run six perturbations
    # ======================================================

    run_perturbation(
        "target",
        "mask",
        mask_region(
            image,
            roles["target_mask"]
        )
    )


    run_perturbation(
        "target",
        "blur",
        blur_region(
            image,
            roles["target_mask"]
        )
    )


    run_perturbation(
        "occluder",
        "mask",
        mask_region(
            image,
            roles["occluder_mask"]
        )
    )


    run_perturbation(
        "occluder",
        "blur",
        blur_region(
            image,
            roles["occluder_mask"]
        )
    )


    run_perturbation(
        "context",
        "mask",
        mask_region(
            image,
            roles["context_mask"]
        )
    )


    run_perturbation(
        "context",
        "blur",
        blur_region(
            image,
            roles["context_mask"]
        )
    )


# ==========================================================
# Save batch results
# ==========================================================

Path("results").mkdir(
    exist_ok=True
)

df = pd.DataFrame(
    results_rows
)

output_csv = (
    "results/batch_results.csv"
)

df.to_csv(
    output_csv,
    index=False
)


print()
print("=" * 70)
print("BATCH COMPLETE")
print("=" * 70)

print(
    f"Total result rows: {len(df)}"
)

print(
    f"Unique targets processed: "
    f"{df['target_ann_id'].nunique() if not df.empty else 0}"
)

print(
    f"Saved to: {output_csv}"
)