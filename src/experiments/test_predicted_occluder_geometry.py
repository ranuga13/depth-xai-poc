import cv2
import numpy as np
from pathlib import Path

from src.datasets.kins_loader import KINSLoader

from src.detector.yolo_segmenter import (
    YOLOSegmenter
)

from src.detector.segmentation_matcher import (
    match_target_to_segment_instance
)

from src.detector.instance_deduplicator import (
    deduplicate_instances,
    calculate_mask_iou
)

from src.detector.target_matcher import (
    calculate_iou,
    xywh_to_xyxy
)

from src.roles.predicted_role_inference import (
    infer_roles_from_predicted_instances
)

from src.roles.scene_roles import (
    find_foreground_occluder,
    polygon_to_mask
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

DEPTH_DIR = Path(
    "results/depth_batch"
)

TARGET_ID = 68

SEGMENTATION_MODEL = (
    "yolo11n-seg.pt"
)

TARGET_MATCH_IOU_THRESHOLD = 0.30

DEDUP_BBOX_IOU_THRESHOLD = 0.80
DEDUP_MASK_IOU_THRESHOLD = 0.70


# Threshold used only for evaluation
OCCLUDER_BBOX_IOU_THRESHOLD = 0.50
OCCLUDER_MASK_IOU_THRESHOLD = 0.50


# ==========================================================
# Load KINS
# ==========================================================

loader = KINSLoader(
    ANNOTATION_PATH
)


target_ann = (
    loader.get_annotation_by_id(
        TARGET_ID
    )
)


if target_ann is None:
    raise RuntimeError(
        f"Target {TARGET_ID} not found."
    )


target_class = (
    loader.get_category_name(
        target_ann["category_id"]
    )
)


image_info = (
    loader.get_image(
        target_ann["image_id"]
    )
)


file_name = (
    image_info["file_name"]
)


image_path = (
    Path(IMAGE_DIR)
    / file_name
)


# ==========================================================
# Load image
# ==========================================================

image = cv2.imread(
    str(image_path)
)


if image is None:
    raise FileNotFoundError(
        image_path
    )


height, width = (
    image.shape[:2]
)


# ==========================================================
# Run YOLO11n-seg
# ==========================================================

segmenter = YOLOSegmenter(
    SEGMENTATION_MODEL
)


instances = (
    segmenter.segment(
        image
    )
)


# ==========================================================
# Remove duplicate instances
# ==========================================================

instances, removed_duplicates = (
    deduplicate_instances(
        instances,
        bbox_iou_threshold=
            DEDUP_BBOX_IOU_THRESHOLD,

        mask_iou_threshold=
            DEDUP_MASK_IOU_THRESHOLD
    )
)


# ==========================================================
# Match target to predicted segmentation
# ==========================================================

target_instance, target_match_iou = (
    match_target_to_segment_instance(
        target_annotation=
            target_ann,

        target_class=
            target_class,

        instances=
            instances,

        min_iou=
            TARGET_MATCH_IOU_THRESHOLD
    )
)


if target_instance is None:
    raise RuntimeError(
        "Could not match target "
        "to predicted segmentation."
    )


# ==========================================================
# Load depth
# ==========================================================

depth_path = (
    DEPTH_DIR
    / f"target_{TARGET_ID}_depth.npy"
)


if not depth_path.exists():
    raise FileNotFoundError(
        depth_path
    )


depth_map = np.load(
    depth_path
)


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
# Fully predicted role inference
# ==========================================================

roles = (
    infer_roles_from_predicted_instances(
        target_instance=
            target_instance,

        instances=
            instances,

        depth_map=
            depth_map,

        height=
            height,

        width=
            width
    )
)


predicted_occluder = (
    roles["predicted_occluder"]
)


# ==========================================================
# KINS reference occluder
# ==========================================================

image_annotations = (
    loader.get_annotations_for_image(
        target_ann["image_id"]
    )
)


kins_occluder = (
    find_foreground_occluder(
        target_annotation=
            target_ann,

        image_annotations=
            image_annotations,

        height=
            height,

        width=
            width
    )
)


if kins_occluder is None:
    raise RuntimeError(
        "No KINS reference occluder found."
    )


# ==========================================================
# Build KINS reference geometry
# ==========================================================

kins_occluder_box = (
    xywh_to_xyxy(
        kins_occluder["i_bbox"]
    )
)


kins_occluder_mask = (
    polygon_to_mask(
        kins_occluder["i_segm"],
        height,
        width
    )
)


kins_occluder_class = (
    loader.get_category_name(
        kins_occluder["category_id"]
    )
)


# ==========================================================
# Evaluate predicted occluder geometry
# ==========================================================

if predicted_occluder is None:

    bbox_iou = 0.0
    mask_iou = 0.0
    same_physical_occluder = False

else:

    predicted_box = (
        predicted_occluder["bbox"]
    )

    predicted_mask = (
        predicted_occluder["mask"]
    )


    bbox_iou = (
        calculate_iou(
            predicted_box,
            kins_occluder_box
        )
    )


    mask_iou = (
        calculate_mask_iou(
            predicted_mask,
            kins_occluder_mask
        )
    )


    # Count as correct if either geometry metric
    # strongly agrees with the KINS occluder.
    same_physical_occluder = (
        bbox_iou
        >= OCCLUDER_BBOX_IOU_THRESHOLD

        or

        mask_iou
        >= OCCLUDER_MASK_IOU_THRESHOLD
    )


# ==========================================================
# Print results
# ==========================================================

print()
print("=" * 90)
print("PREDICTED OCCLUDER GEOMETRY VALIDATION")
print("=" * 90)

print(
    f"Target ID: "
    f"{TARGET_ID}"
)

print(
    f"Image: "
    f"{file_name}"
)

print(
    f"Target class: "
    f"{target_class}"
)

print(
    f"Target segmentation match IoU: "
    f"{target_match_iou:.4f}"
)


print()
print("-" * 90)

print(
    f"KINS occluder ID: "
    f"{kins_occluder['id']}"
)

print(
    f"KINS occluder class: "
    f"{kins_occluder_class}"
)


if predicted_occluder is None:

    print()
    print(
        "Predicted occluder: None"
    )

else:

    print()

    print(
        f"Predicted instance ID: "
        f"{predicted_occluder['instance_id']}"
    )

    print(
        f"Predicted class: "
        f"{predicted_occluder['class_name']}"
    )

    print(
        f"Predicted confidence: "
        f"{predicted_occluder['confidence']:.4f}"
    )


print()
print("-" * 90)

print(
    f"Occluder bbox IoU: "
    f"{bbox_iou:.4f}"
)

print(
    f"Occluder mask IoU: "
    f"{mask_iou:.4f}"
)

print(
    f"Same physical occluder: "
    f"{same_physical_occluder}"
)


print()
print("=" * 90)
print("INTERPRETATION")
print("=" * 90)

if same_physical_occluder:

    print(
        "Prediction matches the KINS foreground "
        "occluder geometrically."
    )

    if (
        predicted_occluder is not None
        and
        predicted_occluder["class_name"]
        != kins_occluder_class
    ):

        print(
            "The semantic class label differs, "
            "but the spatial object is the same."
        )

else:

    print(
        "Prediction does not sufficiently match "
        "the KINS foreground occluder."
    )