import cv2

from datasets.kins_loader import KINSLoader
from detector.yolo_detector import YOLODetector
from detector.target_matcher import match_target


ANNOTATION_PATH = "data/annotations/update_train_2020.json"
IMAGE_PATH = "data/images/training/image_2/006090.png"

TARGET_ANN_ID = 68


loader = KINSLoader(
    ANNOTATION_PATH
)

detector = YOLODetector(
    "yolo11n.pt"
)


target_ann = loader.get_annotation_by_id(
    TARGET_ANN_ID
)

target_class = loader.get_category_name(
    target_ann["category_id"]
)


image = cv2.imread(
    IMAGE_PATH
)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )


detections = detector.detect(
    image
)


matched_detection, iou = match_target(
    target_annotation=target_ann,
    target_class=target_class,
    detections=detections,
    min_iou=0.30
)


if matched_detection is None:
    print("No valid YOLO match found.")
    print(f"Best IoU: {iou:.4f}")

else:
    print("Matched target found")
    print(
        f"Class: "
        f"{matched_detection['class_name']}"
    )
    print(
        f"Confidence: "
        f"{matched_detection['confidence']:.4f}"
    )
    print(
        f"BBox: "
        f"{[round(v, 1) for v in matched_detection['bbox']]}"
    )
    print(
        f"IoU with KINS target: "
        f"{iou:.4f}"
    )