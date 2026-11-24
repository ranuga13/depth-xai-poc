import cv2
import numpy as np
from ultralytics import YOLO


class YOLOSegmenter:
    def __init__(self, model_path="yolo11n-seg.pt"):
        self.model = YOLO(model_path)

    def segment(self, image):
        result = self.model(image, verbose=False)[0]

        instances = []

        if result.boxes is None:
            return instances

        masks = None

        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()

        image_height, image_width = image.shape[:2]

        for index, box in enumerate(result.boxes):
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            bbox = box.xyxy[0].tolist()

            mask = None

            if masks is not None and index < len(masks):
                raw_mask = masks[index]

                mask = cv2.resize(
                    raw_mask,
                    (image_width, image_height),
                    interpolation=cv2.INTER_NEAREST
                )

                mask = mask > 0.5

            instances.append(
                {
                    "instance_id": index,
                    "class_id": class_id,
                    "class_name": self.model.names[class_id],
                    "confidence": confidence,
                    "bbox": bbox,
                    "mask": mask
                }
            )

        return instances