import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ==========================================================
# Configuration
# ==========================================================

INPUT_CSV = "results/batch_results.csv"

OUTPUT_DIR = Path(
    "results/aggregate_analysis"
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
print("BATCH SUMMARY")
print("=" * 70)

print(
    f"Total rows: {len(df)}"
)

print(
    f"Unique targets: "
    f"{df['target_ann_id'].nunique()}"
)

print()


# ==========================================================
# Aggregate by role + perturbation
# ==========================================================

summary = (
    df.groupby(
        [
            "role",
            "perturbation_type"
        ]
    )
    .agg(
        num_samples=(
            "target_ann_id",
            "count"
        ),

        survival_rate=(
            "survived",
            "mean"
        ),

        mean_confidence_change=(
            "confidence_change",
            "mean"
        ),

        median_confidence_change=(
            "confidence_change",
            "median"
        ),

        mean_bbox_iou=(
            "bbox_iou",
            "mean"
        )
    )
    .reset_index()
)


summary["survival_rate"] = (
    summary["survival_rate"] * 100
)


print("=" * 70)
print("ROLE + PERTURBATION SUMMARY")
print("=" * 70)

print(summary)


# ==========================================================
# Aggregate by role only
# ==========================================================

role_summary = (
    df.groupby(
        "role"
    )
    .agg(
        num_samples=(
            "target_ann_id",
            "count"
        ),

        survival_rate=(
            "survived",
            "mean"
        ),

        mean_confidence_change=(
            "confidence_change",
            "mean"
        ),

        median_confidence_change=(
            "confidence_change",
            "median"
        ),

        mean_bbox_iou=(
            "bbox_iou",
            "mean"
        )
    )
    .reset_index()
)


role_summary["survival_rate"] = (
    role_summary["survival_rate"] * 100
)


print()
print("=" * 70)
print("ROLE SUMMARY")
print("=" * 70)

print(role_summary)


# ==========================================================
# Save CSV summaries
# ==========================================================

summary.to_csv(
    OUTPUT_DIR / "role_perturbation_summary.csv",
    index=False
)

role_summary.to_csv(
    OUTPUT_DIR / "role_summary.csv",
    index=False
)


# ==========================================================
# Plot 1: Mean confidence change
# ==========================================================

pivot_conf = summary.pivot(
    index="role",
    columns="perturbation_type",
    values="mean_confidence_change"
)

ax = pivot_conf.plot(
    kind="bar",
    figsize=(9, 6)
)

ax.set_title(
    "Mean Confidence Change by Scene Role"
)

ax.set_xlabel(
    "Scene Role"
)

ax.set_ylabel(
    "Mean Confidence Change"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "mean_confidence_change.png",
    dpi=150
)

plt.close()


# ==========================================================
# Plot 2: Survival rate
# ==========================================================

pivot_survival = summary.pivot(
    index="role",
    columns="perturbation_type",
    values="survival_rate"
)

ax = pivot_survival.plot(
    kind="bar",
    figsize=(9, 6)
)

ax.set_title(
    "Detection Survival Rate by Scene Role"
)

ax.set_xlabel(
    "Scene Role"
)

ax.set_ylabel(
    "Survival Rate (%)"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "survival_rate.png",
    dpi=150
)

plt.close()


# ==========================================================
# Plot 3: Mean bbox IoU
# ==========================================================

pivot_iou = summary.pivot(
    index="role",
    columns="perturbation_type",
    values="mean_bbox_iou"
)

ax = pivot_iou.plot(
    kind="bar",
    figsize=(9, 6)
)

ax.set_title(
    "Mean Bounding Box IoU After Perturbation"
)

ax.set_xlabel(
    "Scene Role"
)

ax.set_ylabel(
    "Mean Bounding Box IoU"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "mean_bbox_iou.png",
    dpi=150
)

plt.close()


print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"Saved analysis to: {OUTPUT_DIR}"
)