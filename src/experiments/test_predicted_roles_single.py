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
    deduplicate_instances
)

from src.roles.predicted_role_inference import (
    infer_roles_from_predicted_instances
)

from src.roles.scene_roles import (
    find_foreground_occluder
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

TARGET_MATCH_IOU_THRESHOLD = (
    0.30
)

DEDUP_BBOX_IOU_THRESHOLD = (
    0.80
)

DEDUP_MASK_IOU_THRESHOLD = (
    0.70
)


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
        f"Target annotation "
        f"{TARGET_ID} not found."
    )


target_class = (
    loader.get_category_name(
        target_ann[
            "category_id"
        ]
    )
)


image_info = (
    loader.get_image(
        target_ann[
            "image_id"
        ]
    )
)


if image_info is None:
    raise RuntimeError(
        "Image information not found."
    )


file_name = (
    image_info[
        "file_name"
    ]
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


original_instance_count = (
    len(instances)
)


# ==========================================================
# Remove duplicate predicted instances
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


print()
print("=" * 90)
print("SEGMENTATION PREPROCESSING")
print("=" * 90)

print(
    f"Original instances: "
    f"{original_instance_count}"
)

print(
    f"Instances after deduplication: "
    f"{len(instances)}"
)

print(
    f"Duplicates removed: "
    f"{len(removed_duplicates)}"
)


if removed_duplicates:

    print()

    print(
        "Removed duplicate instances:"
    )

    for item in (
        removed_duplicates
    ):

        removed_instance = (
            item[
                "instance"
            ]
        )

        duplicate_of = (
            item[
                "duplicate_of"
            ]
        )

        print(
            f"  ID "
            f"{removed_instance['instance_id']} "
            f"({removed_instance['class_name']}) "
            f"-> duplicate of ID "
            f"{duplicate_of['instance_id']} "
            f"({duplicate_of['class_name']}) "
            f"| bbox IoU="
            f"{item['bbox_iou']:.4f} "
            f"| mask IoU="
            f"{item['mask_iou']:.4f}"
        )


# ==========================================================
# Match KINS evaluation target to predicted segmentation
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
        "to predicted segmentation instance. "
        f"Best IoU: "
        f"{target_match_iou:.4f}"
    )


# ==========================================================
# Load predicted depth
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
        (
            width,
            height
        ),
        interpolation=
            cv2.INTER_CUBIC
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
    roles[
        "predicted_occluder"
    ]
)


# ==========================================================
# Predicted occluder information
# ==========================================================

if predicted_occluder is None:

    predicted_instance_id = (
        None
    )

    predicted_class = (
        None
    )

    predicted_confidence = (
        None
    )

    predicted_bbox = (
        None
    )

else:

    predicted_instance_id = (
        predicted_occluder[
            "instance_id"
        ]
    )

    predicted_class = (
        predicted_occluder[
            "class_name"
        ]
    )

    predicted_confidence = (
        predicted_occluder[
            "confidence"
        ]
    )

    predicted_bbox = (
        predicted_occluder[
            "bbox"
        ]
    )


# ==========================================================
# KINS reference occluder
#
# IMPORTANT:
# KINS is used only for evaluation here.
# It is NOT used in predicted role inference.
# ==========================================================

image_annotations = (
    loader.get_annotations_for_image(
        target_ann[
            "image_id"
        ]
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


if kins_occluder is not None:

    kins_occluder_id = (
        kins_occluder[
            "id"
        ]
    )

    kins_occluder_class = (
        loader.get_category_name(
            kins_occluder[
                "category_id"
            ]
        )
    )

else:

    kins_occluder_id = (
        None
    )

    kins_occluder_class = (
        None
    )


# ==========================================================
# Print main results
# ==========================================================

print()
print("=" * 90)
print("FULLY PREDICTED ROLE INFERENCE TEST")
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
    f"Matched segmentation instance: "
    f"{target_instance['instance_id']}"
)

print(
    f"Target segmentation confidence: "
    f"{target_instance['confidence']:.4f}"
)

print(
    f"Target match IoU: "
    f"{target_match_iou:.4f}"
)

print(
    f"Target depth: "
    f"{roles['target_depth']:.4f}"
)


print()


print(
    f"Predicted occluder instance: "
    f"{predicted_instance_id}"
)

print(
    f"Predicted occluder class: "
    f"{predicted_class}"
)


if predicted_confidence is not None:

    print(
        f"Predicted occluder confidence: "
        f"{predicted_confidence:.4f}"
    )


print(
    f"KINS reference occluder: "
    f"{kins_occluder_id} "
    f"({kins_occluder_class})"
)


# ==========================================================
# Predicted candidates
# ==========================================================

print()
print("=" * 90)
print("PREDICTED CANDIDATES")
print("=" * 90)


print(
    f"{'Inst':>8}"
    f"{'Class':>12}"
    f"{'Conf':>10}"
    f"{'Overlap':>12}"
    f"{'Depth':>12}"
    f"{'Depth Diff':>14}"
    f"{'Closer':>10}"
    f"{'Score':>12}"
)


print(
    "-" * 90
)


for candidate in (
    roles[
        "candidates"
    ]
):

    print(
        f"{candidate['instance_id']:>8}"
        f"{candidate['class_name']:>12}"
        f"{candidate['confidence']:>10.4f}"
        f"{candidate['overlap_ratio']:>12.4f}"
        f"{candidate['candidate_depth']:>12.4f}"
        f"{candidate['depth_difference']:>14.4f}"
        f"{str(candidate['is_closer']):>10}"
        f"{candidate['score']:>12.4f}"
    )


# ==========================================================
# Role-mask sizes
# ==========================================================

print()

print(
    f"Target pixels: "
    f"{int(roles['target_mask'].sum())}"
)

print(
    f"Occluder pixels: "
    f"{int(roles['occluder_mask'].sum())}"
)

print(
    f"Context pixels: "
    f"{int(roles['context_mask'].sum())}"
)


# ==========================================================
# Selected occluder details
# ==========================================================

if predicted_occluder is not None:

    print()
    print("=" * 90)
    print("SELECTED OCCLUDER DETAILS")
    print("=" * 90)

    rounded_bbox = [
        round(
            value,
            1
        )
        for value in predicted_bbox
    ]

    print(
        f"Instance ID: "
        f"{predicted_instance_id}"
    )

    print(
        f"Class: "
        f"{predicted_class}"
    )

    print(
        f"Confidence: "
        f"{predicted_confidence:.4f}"
    )

    print(
        f"Bounding box: "
        f"{rounded_bbox}"
    )

else:

    print()
    print(
        "No foreground occluder "
        "was predicted."
    )