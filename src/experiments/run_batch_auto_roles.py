import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from src.datasets.kins_loader import KINSLoader
from src.detector.yolo_detector import YOLODetector
from src.detector.target_matcher import match_target
from src.roles.depth_role_inference import infer_scene_roles_from_depth
from src.perturbation.perturb import mask_region, blur_region
from src.metrics.detection_metrics import calculate_detection_metrics
from src.experiments.experiment_utils import build_result_record


ANNOTATION_PATH = "data/annotations/update_train_2020.json"
IMAGE_DIR = "data/images/training/image_2"

VALIDATED_RESULTS_PATH = "results/batch_results_validated.csv"

DEPTH_DIR = Path(
    "results/depth_batch"
)

OUTPUT_PATH = Path(
    "results/batch_results_auto_roles.csv"
)

MODEL_PATH = "yolo11n.pt"

MATCH_IOU_THRESHOLD = 0.30


loader = KINSLoader(
    ANNOTATION_PATH
)

detector = YOLODetector(
    MODEL_PATH
)

validated_df = pd.read_csv(
    VALIDATED_RESULTS_PATH
)

TARGET_IDS = sorted(
    validated_df[
        "target_ann_id"
    ].unique().tolist()
)

print(
    f"Validated targets: {len(TARGET_IDS)}"
)


rows = []


for index, target_id in enumerate(
    TARGET_IDS
):

    print()
    print("=" * 80)
    print(
        f"TARGET {index + 1}/{len(TARGET_IDS)} "
        f"| ID {target_id}"
    )
    print("=" * 80)


    target_ann = loader.get_annotation_by_id(
        target_id
    )

    if target_ann is None:
        continue


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
        continue


    height, width = image.shape[:2]


    # ======================================================
    # Original YOLO detection
    # ======================================================

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

        print(
            "Original target not matched."
        )

        continue


    # ======================================================
    # Load depth
    # ======================================================

    depth_path = (
        DEPTH_DIR
        / f"target_{target_id}_depth.npy"
    )


    if not depth_path.exists():

        print(
            "Depth map missing."
        )

        continue


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


    # ======================================================
    # Automatic roles
    # ======================================================

    image_annotations = (
        loader.get_annotations_for_image(
            target_ann["image_id"]
        )
    )


    roles = (
        infer_scene_roles_from_depth(
            target_annotation=target_ann,
            image_annotations=image_annotations,
            depth_map=depth_map,
            height=height,
            width=width
        )
    )


    predicted_occluder = (
        roles[
            "predicted_occluder"
        ]
    )


    if predicted_occluder is not None:

        predicted_occluder_id = (
            predicted_occluder[
                "id"
            ]
        )

        predicted_occluder_class = (
            loader.get_category_name(
                predicted_occluder[
                    "category_id"
                ]
            )
        )

    else:

        predicted_occluder_id = None
        predicted_occluder_class = None


    print(
        f"Predicted occluder: "
        f"{predicted_occluder_id} "
        f"({predicted_occluder_class})"
    )


    role_masks = {
        "target":
            roles[
                "target_mask"
            ],

        "occluder":
            roles[
                "occluder_mask"
            ],

        "context":
            roles[
                "context_mask"
            ]
    }


    # ======================================================
    # Perturb each role
    # ======================================================

    for role_name, mask in (
        role_masks.items()
    ):

        for perturbation_type in [
            "mask",
            "blur"
        ]:

            if perturbation_type == "mask":

                perturbed_image = (
                    mask_region(
                        image,
                        mask
                    )
                )

            else:

                perturbed_image = (
                    blur_region(
                        image,
                        mask
                    )
                )


            perturbed_detections = (
                detector.detect(
                    perturbed_image
                )
            )


            perturbed_match, _ = (
                match_target(
                    target_annotation=target_ann,
                    target_class=target_class,
                    detections=perturbed_detections,
                    min_iou=MATCH_IOU_THRESHOLD
                )
            )


            metrics = (
                calculate_detection_metrics(
                    original_detection=original_match,
                    perturbed_detection=perturbed_match
                )
            )


            record = {
                "image_id": target_ann["image_id"],
                "file_name": file_name,

                "target_ann_id": target_ann["id"],
                "target_class": target_class,

                "occluder_ann_id": (
                    predicted_occluder_id
                    if predicted_occluder_id is not None
                    else None
                ),

                "occluder_class": (
                    predicted_occluder_class
                ),

                "occlusion_ratio": (
                    loader.calculate_occlusion_ratio(
                        target_ann
                    )
                ),

                "role": role_name,

                "perturbation_type": (
                    perturbation_type
                ),

                "original_confidence": (
                    metrics[
                        "original_confidence"
                    ]
                ),

                "perturbed_confidence": (
                    metrics[
                        "perturbed_confidence"
                    ]
                ),

                "confidence_change": (
                    metrics[
                        "confidence_change"
                    ]
                ),

                "bbox_iou": (
                    metrics[
                        "bbox_iou"
                    ]
                ),

                "survived": (
                    metrics[
                        "survived"
                    ]
                ),

                "original_match_iou": (
                    original_match_iou
                )
            }


            rows.append(
                record
            )


            print(
                f"{role_name:10} "
                f"{perturbation_type:5} "
                f"| survived="
                f"{metrics['survived']} "
                f"| delta="
                f"{metrics['confidence_change']}"
            )


# ==========================================================
# Save
# ==========================================================

result_df = pd.DataFrame(
    rows
)

result_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print()
print("=" * 80)
print("AUTO-ROLE PERTURBATION COMPLETE")
print("=" * 80)

print(
    f"Rows: {len(result_df)}"
)

print(
    f"Targets: "
    f"{result_df['target_ann_id'].nunique()}"
)

print(
    f"Saved to: "
    f"{OUTPUT_PATH}"
)