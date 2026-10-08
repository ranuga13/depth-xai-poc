import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


INPUT_CSV = "results/batch_results.csv"

OUTPUT_DIR = Path(
    "results/per_target_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# Load results
# ==========================================================

df = pd.read_csv(
    INPUT_CSV
)

print("=" * 70)
print("PER-TARGET ANALYSIS")
print("=" * 70)

print(
    f"Unique targets: "
    f"{df['target_ann_id'].nunique()}"
)

print()


# ==========================================================
# Save a clean per-target table
# ==========================================================

columns = [
    "target_ann_id",
    "file_name",
    "occlusion_ratio",
    "role",
    "perturbation_type",
    "original_confidence",
    "perturbed_confidence",
    "confidence_change",
    "bbox_iou",
    "survived"
]

per_target_table = df[
    columns
].copy()

per_target_table.to_csv(
    OUTPUT_DIR / "per_target_results.csv",
    index=False
)


# ==========================================================
# Confidence-change pivot table
# ==========================================================

confidence_pivot = df.pivot_table(
    index="target_ann_id",
    columns=[
        "role",
        "perturbation_type"
    ],
    values="confidence_change"
)

print(
    "Confidence change by target:"
)

print(
    confidence_pivot
)

confidence_pivot.to_csv(
    OUTPUT_DIR
    / "confidence_change_by_target.csv"
)


# ==========================================================
# Plot each target individually
# ==========================================================

target_ids = (
    df["target_ann_id"]
    .drop_duplicates()
    .tolist()
)


for target_id in target_ids:

    target_df = df[
        df["target_ann_id"]
        == target_id
    ].copy()

    labels = (
        target_df["role"]
        + "-"
        + target_df["perturbation_type"]
    )

    values = target_df[
        "confidence_change"
    ]

    plt.figure(
        figsize=(9, 5)
    )

    plt.bar(
        labels,
        values
    )

    plt.axhline(
        y=0,
        linewidth=1
    )

    plt.title(
        f"Target {target_id} - Confidence Change"
    )

    plt.xlabel(
        "Perturbation"
    )

    plt.ylabel(
        "Confidence Change"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / f"target_{target_id}_confidence_change.png",
        dpi=150
    )

    plt.close()


# ==========================================================
# Combined mask comparison
# ==========================================================

mask_df = df[
    df["perturbation_type"]
    == "mask"
].copy()

mask_pivot = mask_df.pivot(
    index="target_ann_id",
    columns="role",
    values="confidence_change"
)

ax = mask_pivot.plot(
    kind="bar",
    figsize=(11, 6)
)

ax.axhline(
    y=0,
    linewidth=1
)

ax.set_title(
    "Mask Perturbation Effect by Target"
)

ax.set_xlabel(
    "Target Annotation ID"
)

ax.set_ylabel(
    "Confidence Change"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "mask_comparison_all_targets.png",
    dpi=150
)

plt.close()


# ==========================================================
# Combined blur comparison
# ==========================================================

blur_df = df[
    df["perturbation_type"]
    == "blur"
].copy()

blur_pivot = blur_df.pivot(
    index="target_ann_id",
    columns="role",
    values="confidence_change"
)

ax = blur_pivot.plot(
    kind="bar",
    figsize=(11, 6)
)

ax.axhline(
    y=0,
    linewidth=1
)

ax.set_title(
    "Blur Perturbation Effect by Target"
)

ax.set_xlabel(
    "Target Annotation ID"
)

ax.set_ylabel(
    "Confidence Change"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "blur_comparison_all_targets.png",
    dpi=150
)

plt.close()


print()
print("=" * 70)
print("PER-TARGET ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"Saved to: {OUTPUT_DIR}"
)