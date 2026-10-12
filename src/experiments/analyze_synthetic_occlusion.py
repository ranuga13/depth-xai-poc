import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ==========================================================
# Configuration
# ==========================================================

INPUT_CSV = "results/synthetic_occlusion_results.csv"

OUTPUT_DIR = Path(
    "results/synthetic_occlusion_analysis"
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
print("SYNTHETIC OCCLUSION ANALYSIS")
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
# Aggregate by occlusion level
# ==========================================================

summary = (
    df.groupby(
        "synthetic_occlusion_percent"
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
print("OCCLUSION LEVEL SUMMARY")
print("=" * 70)

print(
    summary.to_string(
        index=False
    )
)


# ==========================================================
# Save summary
# ==========================================================

summary.to_csv(
    OUTPUT_DIR
    / "synthetic_occlusion_summary.csv",
    index=False
)


# ==========================================================
# Plot 1: Survival rate
# ==========================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    summary[
        "synthetic_occlusion_percent"
    ],
    summary[
        "survival_rate"
    ],
    marker="o"
)

plt.title(
    "Detection Survival vs Synthetic Occlusion"
)

plt.xlabel(
    "Synthetic Occlusion (%)"
)

plt.ylabel(
    "Survival Rate (%)"
)

plt.ylim(
    0,
    105
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "survival_vs_occlusion.png",
    dpi=150
)

plt.close()


# ==========================================================
# Plot 2: Mean confidence change
# ==========================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    summary[
        "synthetic_occlusion_percent"
    ],
    summary[
        "mean_confidence_change"
    ],
    marker="o"
)

plt.axhline(
    y=0,
    linewidth=1
)

plt.title(
    "Mean Confidence Change vs Synthetic Occlusion"
)

plt.xlabel(
    "Synthetic Occlusion (%)"
)

plt.ylabel(
    "Mean Confidence Change"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "mean_confidence_vs_occlusion.png",
    dpi=150
)

plt.close()


# ==========================================================
# Plot 3: Median confidence change
# ==========================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    summary[
        "synthetic_occlusion_percent"
    ],
    summary[
        "median_confidence_change"
    ],
    marker="o"
)

plt.axhline(
    y=0,
    linewidth=1
)

plt.title(
    "Median Confidence Change vs Synthetic Occlusion"
)

plt.xlabel(
    "Synthetic Occlusion (%)"
)

plt.ylabel(
    "Median Confidence Change"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "median_confidence_vs_occlusion.png",
    dpi=150
)

plt.close()


# ==========================================================
# Plot 4: Mean bbox IoU
# ==========================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    summary[
        "synthetic_occlusion_percent"
    ],
    summary[
        "mean_bbox_iou"
    ],
    marker="o"
)

plt.title(
    "Localization Stability vs Synthetic Occlusion"
)

plt.xlabel(
    "Synthetic Occlusion (%)"
)

plt.ylabel(
    "Mean Bounding Box IoU"
)

plt.ylim(
    0,
    1
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "bbox_iou_vs_occlusion.png",
    dpi=150
)

plt.close()


# ==========================================================
# Per-target pivot table
# ==========================================================

confidence_pivot = df.pivot_table(
    index="target_ann_id",
    columns="synthetic_occlusion_percent",
    values="confidence_change"
)

confidence_pivot.to_csv(
    OUTPUT_DIR
    / "confidence_change_by_target.csv"
)


survival_pivot = df.pivot_table(
    index="target_ann_id",
    columns="synthetic_occlusion_percent",
    values="survived"
)

survival_pivot.to_csv(
    OUTPUT_DIR
    / "survival_by_target.csv"
)


bbox_pivot = df.pivot_table(
    index="target_ann_id",
    columns="synthetic_occlusion_percent",
    values="bbox_iou"
)

bbox_pivot.to_csv(
    OUTPUT_DIR
    / "bbox_iou_by_target.csv"
)


# ==========================================================
# Count disappeared detections
# ==========================================================

failure_summary = (
    df.groupby(
        "synthetic_occlusion_percent"
    )["survived"]
    .apply(
        lambda values:
        (~values.astype(bool)).sum()
    )
    .reset_index(
        name="num_detection_failures"
    )
)


failure_summary.to_csv(
    OUTPUT_DIR
    / "detection_failures_by_occlusion.csv",
    index=False
)


print()
print("=" * 70)
print("DETECTION FAILURES")
print("=" * 70)

print(
    failure_summary.to_string(
        index=False
    )
)


# ==========================================================
# Finish
# ==========================================================

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"Saved to: {OUTPUT_DIR}"
)