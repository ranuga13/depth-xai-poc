from ultralytics import YOLO


class YOLODetector:
    def __init__(self, model_path="yolo11n.pt"):
        self.model = YOLO(model_path)

    def detect(self, image):
        results = self.model(
            image,
            verbose=False
        )[0]

        detections = []

        for box in results.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            bbox = box.xyxy[0].tolist()

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": self.model.names[class_id],
                    "confidence": confidence,
                    "bbox": bbox
                }
            )

        return detections