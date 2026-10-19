import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from src.datasets.kins_loader import KINSLoader
from src.roles.scene_roles import build_scene_roles


# ==========================================================
# Configuration
# ==========================================================

ANNOTATION_PATH = "data/annotations/update_train_2020.json"
IMAGE_DIR = "data/images/training/image_2"

DRISE_DIR = Path(
    "results/drise_full"
)

OUTPUT_DIR = Path(
    "results/drise_role_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VALIDATED_RESULTS_PATH = (
    "results/batch_results_validated.csv"
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


# ==========================================================
# Setup
# ==========================================================

loader = KINSLoader(
    ANNOTATION_PATH
)

results_rows = []


# ==========================================================
# Helper
# ==========================================================

def analyze_region(
    role_name,
    mask,
    saliency
):
    pixel_count = int(
        mask.sum()
    )

    if pixel_count == 0:
        return {
            "role": role_name,
            "pixel_count": 0,
            "mean_saliency": np.nan,
            "total_saliency": 0.0
        }

    values = saliency[
        mask
    ]

    return {
        "role": role_name,
        "pixel_count": pixel_count,
        "mean_saliency": float(
            values.mean()
        ),
        "total_saliency": float(
            values.sum()
        )
    }


# ==========================================================
# Process targets
# ==========================================================

for target_id in TARGET_IDS:

    target_ann = loader.get_annotation_by_id(
        target_id
    )

    if target_ann is None:
        print(
            f"Skipping {target_id}: annotation not found"
        )
        continue


    image_info = loader.get_image(
        target_ann["image_id"]
    )

    if image_info is None:
        print(
            f"Skipping {target_id}: image info not found"
        )
        continue


    file_name = image_info["file_name"]

    image_path = (
        Path(IMAGE_DIR)
        / file_name
    )


    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        print(
            f"Skipping {target_id}: image could not be loaded"
        )
        continue


    height, width = image.shape[:2]


    # ======================================================
    # Load saliency map
    # ======================================================

    saliency_path = (
        DRISE_DIR
        / f"target_{target_id}_drise.npy"
    )

    if not saliency_path.exists():
        print(
            f"Skipping {target_id}: saliency map missing"
        )
        continue


    saliency = np.load(
        saliency_path
    )


    # ======================================================
    # Build semantic roles
    # ======================================================

    image_annotations = (
        loader.get_annotations_for_image(
            target_ann["image_id"]
        )
    )


    roles = build_scene_roles(
        target_annotation=target_ann,
        image_annotations=image_annotations,
        height=height,
        width=width
    )


    role_results = []


    role_results.append(
        analyze_region(
            "target",
            roles["target_mask"],
            saliency
        )
    )


    role_results.append(
        analyze_region(
            "occluder",
            roles["occluder_mask"],
            saliency
        )
    )


    role_results.append(
        analyze_region(
            "context",
            roles["context_mask"],
            saliency
        )
    )


    # ======================================================
    # Total saliency across three roles
    # ======================================================

    total_role_saliency = sum(
        item["total_saliency"]
        for item in role_results
    )


    for item in role_results:

        if total_role_saliency > 0:

            role_percent = (
                item["total_saliency"]
                / total_role_saliency
                * 100
            )

        else:
            role_percent = 0.0


        results_rows.append(
            {
                "target_ann_id": target_id,
                "file_name": file_name,
                "role": item["role"],
                "pixel_count": item[
                    "pixel_count"
                ],
                "mean_saliency": item[
                    "mean_saliency"
                ],
                "total_saliency": item[
                    "total_saliency"
                ],
                "role_saliency_percent": (
                    role_percent
                )
            }
        )


    # ======================================================
    # Print per target
    # ======================================================

    valid_roles = [
        item
        for item in role_results
        if not np.isnan(
            item["mean_saliency"]
        )
    ]


    best_role = max(
        valid_roles,
        key=lambda item:
        item["mean_saliency"]
    )


    print()
    print("=" * 70)
    print(
        f"TARGET {target_id}"
    )
    print("=" * 70)

    for item in role_results:

        print(
            f"{item['role']:10} | "
            f"mean={item['mean_saliency']:.4f} | "
            f"total={item['total_saliency']:.2f}"
        )

    print(
        f"Highest mean-saliency role: "
        f"{best_role['role']}"
    )


# ==========================================================
# Save detailed results
# ==========================================================

df = pd.DataFrame(
    results_rows
)


detail_csv = (
    OUTPUT_DIR
    / "drise_role_results.csv"
)

df.to_csv(
    detail_csv,
    index=False
)


# ==========================================================
# Create ranking summary
# ==========================================================

pivot = df.pivot_table(
    index="target_ann_id",
    columns="role",
    values="mean_saliency"
).reset_index()


def find_best_role(row):

    values = {
        "target": row.get(
            "target",
            np.nan
        ),

        "occluder": row.get(
            "occluder",
            np.nan
        ),

        "context": row.get(
            "context",
            np.nan
        )
    }

    values = {
        key: value
        for key, value in values.items()
        if not pd.isna(value)
    }

    return max(
        values,
        key=values.get
    )


pivot[
    "highest_saliency_role"
] = pivot.apply(
    find_best_role,
    axis=1
)


ranking_csv = (
    OUTPUT_DIR
    / "drise_role_ranking.csv"
)


pivot.to_csv(
    ranking_csv,
    index=False
)


print()
print("=" * 70)
print("D-RISE ROLE ANALYSIS COMPLETE")
print("=" * 70)

print(
    pivot.to_string(
        index=False
    )
)

print()

print(
    f"Saved detailed results to: "
    f"{detail_csv}"
)

print(
    f"Saved rankings to: "
    f"{ranking_csv}"
)