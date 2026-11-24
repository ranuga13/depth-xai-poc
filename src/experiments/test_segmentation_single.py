import cv2

from src.detector.yolo_segmenter import YOLOSegmenter


IMAGE_PATH = (
    "data/images/training/image_2/006090.png"
)

MODEL_PATH = "yolo11n-seg.pt"


image = cv2.imread(
    IMAGE_PATH
)

if image is None:
    raise FileNotFoundError(
        IMAGE_PATH
    )


segmenter = YOLOSegmenter(
    MODEL_PATH
)


instances = segmenter.segment(
    image
)


print()
print("=" * 80)
print("YOLO11N-SEG INSTANCE SEGMENTATION TEST")
print("=" * 80)

print(
    f"Instances detected: {len(instances)}"
)

print()


for instance in instances:

    mask_pixels = (
        int(instance["mask"].sum())
        if instance["mask"] is not None
        else 0
    )

    print(
        f"ID={instance['instance_id']:>2} "
        f"class={instance['class_name']:<12} "
        f"conf={instance['confidence']:.4f} "
        f"bbox="
        f"{[round(v, 1) for v in instance['bbox']]} "
        f"mask_pixels={mask_pixels}"
    )