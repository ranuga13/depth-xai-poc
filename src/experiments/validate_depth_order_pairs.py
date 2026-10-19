import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from src.datasets.kins_loader import KINSLoader
from src.roles.scene_roles import polygon_to_mask


ANNOTATION_PATH = "data/annotations/update_train_2020.json"

IMAGE_DIR = "data/images/training/image_2"

VALIDATED_RESULTS_PATH = (
    "results/batch_results_validated.csv"
)

DEPTH_DIR = Path(
    "results/depth_batch"
)

OUTPUT_DIR = Path(
    "results/depth_order_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


loader = KINSLoader(
    ANNOTATION_PATH
)

validated_df = pd.read_csv(
    VALIDATED_RESULTS_PATH
)

target_ids = sorted(
    validated_df[
        "target_ann_id"
    ].unique().tolist()
)


def median_depth(
    mask,
    depth_map
):
    values = depth_map[
        mask
    ]

    if values.size == 0:
        return np.nan

    return float(
        np.median(values)
    )


rows = []


for target_id in target_ids:

    target_rows = validated_df[
        validated_df[
            "target_ann_id"
        ] == target_id
    ]

    if target_rows.empty:
        continue

    target_row = target_rows.iloc[0]

    occluder_id = target_row[
        "occluder_ann_id"
    ]

    if pd.isna(
        occluder_id
    ):
        continue

    occluder_id = int(
        occluder_id
    )


    target_ann = loader.get_annotation_by_id(
        target_id
    )

    occluder_ann = loader.get_annotation_by_id(
        occluder_id
    )


    image_info = loader.get_image(
        target_ann[
            "image_id"
        ]
    )

    file_name = image_info[
        "file_name"
    ]

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


    depth_path = (
        DEPTH_DIR
        / f"target_{target_id}_depth.npy"
    )

    if not depth_path.exists():

        print(
            f"Missing depth map for "
            f"target {target_id}"
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


    # Target visible mask
    target_mask = polygon_to_mask(
        target_ann[
            "i_segm"
        ],
        height,
        width
    )


    # Known KINS occluder visible mask
    occluder_mask = polygon_to_mask(
        occluder_ann[
            "i_segm"
        ],
        height,
        width
    )


    target_depth = median_depth(
        target_mask,
        depth_map
    )

    occluder_depth = median_depth(
        occluder_mask,
        depth_map
    )


    depth_difference = (
        occluder_depth
        - target_depth
    )


    # Based on our Depth Anything V2 outputs:
    # larger value = closer
    correct_order = (
        depth_difference > 0
    )


    rows.append(
        {
            "target_ann_id":
                target_id,

            "occluder_ann_id":
                occluder_id,

            "file_name":
                file_name,

            "target_depth":
                target_depth,

            "occluder_depth":
                occluder_depth,

            "depth_difference":
                depth_difference,

            "correct_depth_order":
                correct_order
        }
    )


result_df = pd.DataFrame(
    rows
)


output_path = (
    OUTPUT_DIR
    / "depth_order_validation.csv"
)


result_df.to_csv(
    output_path,
    index=False
)


total = len(
    result_df
)

correct_count = int(
    result_df[
        "correct_depth_order"
    ].sum()
)


accuracy = (
    correct_count
    / total
    * 100
    if total > 0
    else 0.0
)


print()
print("=" * 80)
print("INDEPENDENT DEPTH ORDER VALIDATION")
print("=" * 80)

print(
    f"Total pairs: "
    f"{total}"
)

print(
    f"Correct depth ordering: "
    f"{correct_count}"
)

print(
    f"Incorrect depth ordering: "
    f"{total - correct_count}"
)

print(
    f"Depth-order accuracy: "
    f"{accuracy:.2f}%"
)

print()
print(
    f"Saved to: "
    f"{output_path}"
)


wrong_df = result_df[
    result_df[
        "correct_depth_order"
    ] == False
]


if not wrong_df.empty:

    print()
    print(
        "Incorrect pairs:"
    )

    print(
        wrong_df[
            [
                "target_ann_id",
                "occluder_ann_id",
                "target_depth",
                "occluder_depth",
                "depth_difference"
            ]
        ].to_string(
            index=False
        )
    )