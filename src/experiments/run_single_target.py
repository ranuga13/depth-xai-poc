import cv2
import pandas as pd
from pathlib import Path

from src.experiments.experiment_utils import build_result_record
from src.datasets.kins_loader import KINSLoader
from src.detector.yolo_detector import YOLODetector
from src.detector.target_matcher import match_target
from src.roles.scene_roles import build_scene_roles
from src.perturbation.perturb import mask_region, blur_region
from src.metrics.detection_metrics import calculate_detection_metrics


# ==========================================================
# Configuration
# ==========================================================

ANNOTATION_PATH = "data/annotations/update_train_2020.json"
IMAGE_PATH = "data/images/training/image_2/006090.png"

TARGET_ANN_ID = 68

MATCH_IOU_THRESHOLD = 0.30


# ==========================================================
# Setup
# ==========================================================

loader = KINSLoader(
    ANNOTATION_PATH
)

detector = YOLODetector(
    "yolo11n.pt"
)


# ==========================================================
# Load target annotation
# ==========================================================

target_ann = loader.get_annotation_by_id(
    TARGET_ANN_ID
)

if target_ann is None:
    raise ValueError(
        f"Annotation {TARGET_ANN_ID} not found"
    )


target_class = loader.get_category_name(
    target_ann["category_id"]
)


image_annotations = loader.get_annotations_for_image(
    target_ann["image_id"]
)


# ==========================================================
# Load image
# ==========================================================

image = cv2.imread(
    IMAGE_PATH
)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )

height, width = image.shape[:2]


# ==========================================================
# Build scene roles
# ==========================================================

roles = build_scene_roles(
    target_annotation=target_ann,
    image_annotations=image_annotations,
    height=height,
    width=width
)

results_rows = []

image_info = loader.get_image(
    target_ann["image_id"]
)

occlusion_ratio = loader.calculate_occlusion_ratio(
    target_ann
)

occluder_ann = roles["occluder_annotation"]

occluder_ann_id = (
    occluder_ann["id"]
    if occluder_ann is not None
    else None
)

occluder_class = (
    loader.get_category_name(
        occluder_ann["category_id"]
    )
    if occluder_ann is not None
    else None
)

print("=" * 60)
print("SCENE ROLES")
print("=" * 60)

print(
    f"Target annotation: "
    f"{target_ann['id']}"
)

if roles["occluder_annotation"] is not None:
    print(
        f"Occluder annotation: "
        f"{roles['occluder_annotation']['id']}"
    )
else:
    print("Occluder annotation: None")


# ==========================================================
# Original YOLO detection
# ==========================================================

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
    raise RuntimeError(
        "Could not match the original YOLO detection "
        "to the KINS target."
    )


print()
print("=" * 60)
print("ORIGINAL TARGET")
print("=" * 60)

print(
    f"Class: "
    f"{original_match['class_name']}"
)

print(
    f"Confidence: "
    f"{original_match['confidence']:.4f}"
)

print(
    f"BBox: "
    f"{[round(v, 1) for v in original_match['bbox']]}"
)

print(
    f"KINS <-> YOLO IoU: "
    f"{original_iou:.4f}"
)


# ==========================================================
# Perturbation experiment helper
# ==========================================================

def run_perturbation(
    role_name,
    perturbation_name,
    perturbed_image
):
    detections = detector.detect(
        perturbed_image
    )

    matched_detection, match_iou = match_target(
        target_annotation=target_ann,
        target_class=target_class,
        detections=detections,
        min_iou=MATCH_IOU_THRESHOLD
    )

    metrics = calculate_detection_metrics(
        original_detection=original_match,
        perturbed_detection=matched_detection
    )

    print()
    print("=" * 60)
    print(
        f"{role_name.upper()} "
        f"- {perturbation_name.upper()}"
    )
    print("=" * 60)

    print(
        f"Survived: "
        f"{metrics['survived']}"
    )

    print(
        f"Original confidence: "
        f"{metrics['original_confidence']:.4f}"
    )

    if metrics["survived"]:

        print(
            f"Perturbed confidence: "
            f"{metrics['perturbed_confidence']:.4f}"
        )

        print(
            f"Confidence change: "
            f"{metrics['confidence_change']:.4f}"
        )

        print(
            f"BBox IoU: "
            f"{metrics['bbox_iou']:.4f}"
        )

        print(
            f"Match IoU: "
            f"{match_iou:.4f}"
        )

    else:
        print(
            "Target could not be matched "
            "after perturbation."
        )

        print(
            f"Best match IoU: "
            f"{match_iou:.4f}"
        )

    record = build_result_record(
    image_id=target_ann["image_id"],
    file_name=image_info["file_name"],
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

    results_rows.append(record)

    return metrics


# ==========================================================
# Target perturbations
# ==========================================================

target_masked = mask_region(
    image,
    roles["target_mask"]
)

run_perturbation(
    role_name="target",
    perturbation_name="mask",
    perturbed_image=target_masked
)


target_blurred = blur_region(
    image,
    roles["target_mask"]
)

run_perturbation(
    role_name="target",
    perturbation_name="blur",
    perturbed_image=target_blurred
)


# ==========================================================
# Occluder perturbations
# ==========================================================

occluder_masked = mask_region(
    image,
    roles["occluder_mask"]
)

run_perturbation(
    role_name="occluder",
    perturbation_name="mask",
    perturbed_image=occluder_masked
)


occluder_blurred = blur_region(
    image,
    roles["occluder_mask"]
)

run_perturbation(
    role_name="occluder",
    perturbation_name="blur",
    perturbed_image=occluder_blurred
)


# ==========================================================
# Context perturbations
# ==========================================================

context_masked = mask_region(
    image,
    roles["context_mask"]
)

run_perturbation(
    role_name="context",
    perturbation_name="mask",
    perturbed_image=context_masked
)


context_blurred = blur_region(
    image,
    roles["context_mask"]
)

run_perturbation(
    role_name="context",
    perturbation_name="blur",
    perturbed_image=context_blurred
)


# ==========================================================
# Save results
# ==========================================================

Path("results").mkdir(
    exist_ok=True
)

df = pd.DataFrame(
    results_rows
)

output_csv = "results/single_target_results.csv"

df.to_csv(
    output_csv,
    index=False
)

print()
print("=" * 60)
print("RESULTS SAVED")
print("=" * 60)

print(df)

print()
print(
    f"Saved to: {output_csv}"
)