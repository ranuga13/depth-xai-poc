import cv2

from src.datasets.kins_loader import KINSLoader
from src.detector.yolo_segmenter import YOLOSegmenter
from src.detector.segmentation_matcher import (
    match_target_to_segment_instance
)


ANNOTATION_PATH = (
    "data/annotations/update_train_2020.json"
)

IMAGE_DIR = (
    "data/images/training/image_2"
)

TARGET_ID = 68


loader = KINSLoader(
    ANNOTATION_PATH
)


target_ann = loader.get_annotation_by_id(
    TARGET_ID
)


target_class = loader.get_category_name(
    target_ann["category_id"]
)


image_info = loader.get_image(
    target_ann["image_id"]
)


image_path = (
    f"{IMAGE_DIR}/"
    f"{image_info['file_name']}"
)


image = cv2.imread(
    image_path
)

if image is None:
    raise FileNotFoundError(
        image_path
    )


segmenter = YOLOSegmenter(
    "yolo11n-seg.pt"
)


instances = segmenter.segment(
    image
)


matched_instance, match_iou = (
    match_target_to_segment_instance(
        target_annotation=target_ann,
        target_class=target_class,
        instances=instances,
        min_iou=0.30
    )
)


print()
print("=" * 80)
print("SEGMENTATION TARGET MATCH TEST")
print("=" * 80)

print(
    f"Target ID: {TARGET_ID}"
)

print(
    f"Target class: {target_class}"
)

print(
    f"Predicted instances: "
    f"{len(instances)}"
)


if matched_instance is None:

    print()
    print(
        "No matching segmentation "
        "instance found."
    )

    print(
        f"Best IoU: {match_iou:.4f}"
    )

else:

    print()
    print(
        f"Matched instance ID: "
        f"{matched_instance['instance_id']}"
    )

    print(
        f"Predicted class: "
        f"{matched_instance['class_name']}"
    )

    print(
        f"Confidence: "
        f"{matched_instance['confidence']:.4f}"
    )

    print(
        f"Match IoU: "
        f"{match_iou:.4f}"
    )

    rounded_bbox = [
        round(v, 1)
        for v in matched_instance["bbox"]
    ]

    print(
        f"Predicted bbox: "
        f"{rounded_bbox}"
    )

    print(
        f"Mask pixels: "
        f"{int(matched_instance['mask'].sum())}"
    )