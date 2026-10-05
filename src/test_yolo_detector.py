import cv2

from detector.yolo_detector import YOLODetector


IMAGE_PATH = "data/images/training/image_2/006090.png"


image = cv2.imread(
    IMAGE_PATH
)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )


detector = YOLODetector(
    "yolo11n.pt"
)


detections = detector.detect(
    image
)


print(
    f"Found {len(detections)} detections"
)

print()


for i, detection in enumerate(detections):

    print(
        f"Detection {i}"
    )

    print(
        f"Class: {detection['class_name']}"
    )

    print(
        f"Confidence: "
        f"{detection['confidence']:.4f}"
    )

    print(
        f"BBox: "
        f"{[round(v, 1) for v in detection['bbox']]}"
    )

    print("-" * 50)