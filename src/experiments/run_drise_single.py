import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.datasets.kins_loader import KINSLoader
from src.detector.yolo_detector import YOLODetector
from src.detector.target_matcher import match_target
from src.detector.target_matcher import calculate_iou


# ==========================================================
# Configuration
# ==========================================================

ANNOTATION_PATH = "data/annotations/update_train_2020.json"
IMAGE_DIR = "data/images/training/image_2"

MODEL_PATH = "yolo11n.pt"

TARGET_ANN_ID = 68

MATCH_IOU_THRESHOLD = 0.30

NUM_MASKS = 500

GRID_SIZE = 8

MASK_PROBABILITY = 0.5

OUTPUT_DIR = Path(
    "results/drise"
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
# Load target
# ==========================================================

target_ann = loader.get_annotation_by_id(
    TARGET_ANN_ID
)

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
    raise FileNotFoundError(
        image_path
    )


height, width = image.shape[:2]


# ==========================================================
# Original target detection
# ==========================================================

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
    raise RuntimeError(
        "Original YOLO target could not be matched."
    )


print("=" * 70)
print("D-RISE SINGLE TARGET")
print("=" * 70)

print(
    f"Target ID: {TARGET_ANN_ID}"
)

print(
    f"Image: {file_name}"
)

print(
    f"Original confidence: "
    f"{original_match['confidence']:.4f}"
)

print(
    f"Original IoU: "
    f"{original_match_iou:.4f}"
)


# ==========================================================
# Helper: generate random mask
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
    ).astype(
        np.float32
    )

    mask = cv2.resize(
        small_mask,
        (width, height),
        interpolation=cv2.INTER_LINEAR
    )

    return mask


# ==========================================================
# Helper: D-RISE similarity
# ==========================================================

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


        bbox_iou = calculate_iou(
            original_bbox,
            detection["bbox"]
        )


        confidence = (
            detection["confidence"]
        )


        score = (
            bbox_iou
            * confidence
        )


        if score > best_score:
            best_score = score


    return best_score


# ==========================================================
# D-RISE
# ==========================================================

saliency = np.zeros(
    (height, width),
    dtype=np.float32
)


for mask_index in range(
    NUM_MASKS
):

    random_mask = (
        generate_random_mask(
            height=height,
            width=width,
            grid_size=GRID_SIZE,
            probability=MASK_PROBABILITY
        )
    )


    masked_image = (
        image.astype(np.float32)
        * random_mask[..., None]
    ).astype(
        np.uint8
    )


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
        (mask_index + 1) % 50
        == 0
    ):
        print(
            f"Processed "
            f"{mask_index + 1}/"
            f"{NUM_MASKS} masks"
        )


# ==========================================================
# Normalize saliency
# ==========================================================

saliency = (
    saliency
    / NUM_MASKS
)


saliency_min = saliency.min()
saliency_max = saliency.max()


if saliency_max > saliency_min:

    saliency = (
        saliency
        - saliency_min
    ) / (
        saliency_max
        - saliency_min
    )


# ==========================================================
# Visualization
# ==========================================================

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


plt.figure(
    figsize=(15, 5)
)


plt.subplot(
    1,
    3,
    1
)

plt.imshow(
    image_rgb
)

plt.title(
    "Original"
)

plt.axis(
    "off"
)


plt.subplot(
    1,
    3,
    2
)

plt.imshow(
    saliency,
    cmap="jet"
)

plt.title(
    "D-RISE Saliency"
)

plt.axis(
    "off"
)


plt.subplot(
    1,
    3,
    3
)

plt.imshow(
    image_rgb
)

plt.imshow(
    saliency,
    cmap="jet",
    alpha=0.5
)

plt.title(
    "D-RISE Overlay"
)

plt.axis(
    "off"
)


plt.tight_layout()


output_path = (
    OUTPUT_DIR
    / f"target_{TARGET_ANN_ID}_drise.png"
)


plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# Save saliency map
# ==========================================================

np.save(
    OUTPUT_DIR
    / f"target_{TARGET_ANN_ID}_drise.npy",
    saliency
)


print()
print("=" * 70)
print("D-RISE COMPLETE")
print("=" * 70)

print(
    f"Saved image: "
    f"{output_path}"
)