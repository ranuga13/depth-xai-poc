import cv2

from src.detector.yolo_segmenter import (
    YOLOSegmenter
)

from src.detector.instance_deduplicator import (
    deduplicate_instances
)


IMAGE_PATH = (
    "data/images/training/image_2/006090.png"
)


image = cv2.imread(
    IMAGE_PATH
)

if image is None:
    raise FileNotFoundError(
        IMAGE_PATH
    )


segmenter = YOLOSegmenter(
    "yolo11n-seg.pt"
)


instances = segmenter.segment(
    image
)


filtered_instances, removed = (
    deduplicate_instances(
        instances,
        bbox_iou_threshold=0.80,
        mask_iou_threshold=0.70
    )
)


print()
print("=" * 90)
print("INSTANCE DEDUPLICATION TEST")
print("=" * 90)

print(
    f"Original instances: "
    f"{len(instances)}"
)

print(
    f"Kept instances: "
    f"{len(filtered_instances)}"
)

print(
    f"Removed duplicates: "
    f"{len(removed)}"
)


print()
print("=" * 90)
print("KEPT")
print("=" * 90)


for instance in filtered_instances:

    print(
        f"ID={instance['instance_id']:>2} "
        f"class={instance['class_name']:<10} "
        f"conf={instance['confidence']:.4f} "
        f"pixels={int(instance['mask'].sum())}"
    )


print()
print("=" * 90)
print("REMOVED")
print("=" * 90)


for item in removed:

    instance = item[
        "instance"
    ]

    duplicate_of = item[
        "duplicate_of"
    ]

    print(
        f"ID={instance['instance_id']:>2} "
        f"({instance['class_name']}) "
        f"removed as duplicate of "
        f"ID={duplicate_of['instance_id']:>2} "
        f"({duplicate_of['class_name']}) "
        f"| bbox IoU="
        f"{item['bbox_iou']:.4f} "
        f"| mask IoU="
        f"{item['mask_iou']:.4f}"
    )