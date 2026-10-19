import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ==========================================================
# Configuration
# ==========================================================

ROLE_RESULTS_CSV = "results/batch_results_validated.csv"
BASELINE_RESULTS_CSV = "results/random_baseline_results.csv"

OUTPUT_DIR = Path(
    "results/baseline_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# Load data
# ==========================================================

role_df = pd.read_csv(
    ROLE_RESULTS_CSV
)

baseline_df = pd.read_csv(
    BASELINE_RESULTS_CSV
)


print("=" * 70)
print("BASELINE COMPARISON")
print("=" * 70)

print(
    f"Role-aware targets: "
    f"{role_df['target_ann_id'].nunique()}"
)

print(
    f"Baseline targets: "
    f"{baseline_df['target_ann_id'].nunique()}"
)

print()


# ==========================================================
# Keep only MASK results from role-aware method
# ==========================================================

role_mask_df = role_df[
    role_df["perturbation_type"] == "mask"
].copy()


# ==========================================================
# Role-aware summary
# ==========================================================

role_summary = (
    role_mask_df
    .groupby("role")
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


role_summary[
    "survival_rate"
] *= 100


role_summary[
    "method"
] = (
    "role_"
    + role_summary["role"]
)


# ==========================================================
# Random baseline summary
# ==========================================================

baseline_summary = pd.DataFrame(
    [
        {
            "role": "random",

            "num_samples": len(
                baseline_df
            ),

            "survival_rate": (
                baseline_df[
                    "survived"
                ].mean()
                * 100
            ),

            "mean_confidence_change": (
                baseline_df[
                    "confidence_change"
                ].mean()
            ),

            "median_confidence_change": (
                baseline_df[
                    "confidence_change"
                ].median()
            ),

            "mean_bbox_iou": (
                baseline_df[
                    "bbox_iou"
                ].mean()
            ),

            "method": (
                "random_baseline"
            )
        }
    ]
)


# ==========================================================
# Combine summaries
# ==========================================================

comparison = pd.concat(
    [
        role_summary,
        baseline_summary
    ],
    ignore_index=True
)


comparison = comparison[
    [
        "method",
        "role",
        "num_samples",
        "survival_rate",
        "mean_confidence_change",
        "median_confidence_change",
        "mean_bbox_iou"
    ]
]


print("=" * 70)
print("OVERALL COMPARISON")
print("=" * 70)

print(
    comparison.to_string(
        index=False
    )
)


comparison.to_csv(
    OUTPUT_DIR
    / "baseline_comparison_summary.csv",
    index=False
)


# ==========================================================
# Per-target baseline averages
# ==========================================================

baseline_per_target = (
    baseline_df
    .groupby(
        "target_ann_id"
    )
    .agg(
        baseline_survival_rate=(
            "survived",
            "mean"
        ),

        baseline_mean_conf_change=(
            "confidence_change",
            "mean"
        ),

        baseline_median_conf_change=(
            "confidence_change",
            "median"
        ),

        baseline_mean_bbox_iou=(
            "bbox_iou",
            "mean"
        )
    )
    .reset_index()
)


baseline_per_target[
    "baseline_survival_rate"
] *= 100


# ==========================================================
# Per-target role-aware mask values
# ==========================================================

role_per_target = role_mask_df.pivot_table(
    index="target_ann_id",
    columns="role",
    values="confidence_change"
).reset_index()


role_per_target = (
    role_per_target.rename(
        columns={
            "target": (
                "target_mask_conf_change"
            ),
            "occluder": (
                "occluder_mask_conf_change"
            ),
            "context": (
                "context_mask_conf_change"
            )
        }
    )
)


# ==========================================================
# Merge per-target comparison
# ==========================================================

per_target_comparison = pd.merge(
    baseline_per_target,
    role_per_target,
    on="target_ann_id",
    how="inner"
)


per_target_comparison.to_csv(
    OUTPUT_DIR
    / "per_target_baseline_comparison.csv",
    index=False
)


print()
print("=" * 70)
print("PER-TARGET COMPARISON")
print("=" * 70)

print(
    per_target_comparison.to_string(
        index=False
    )
)


# ==========================================================
# Plot 1: Mean confidence change
# ==========================================================

plot_df = comparison.copy()


plt.figure(
    figsize=(9, 6)
)

plt.bar(
    plot_df["method"],
    plot_df[
        "mean_confidence_change"
    ]
)

plt.axhline(
    y=0,
    linewidth=1
)

plt.title(
    "Mean Confidence Change: "
    "Role-Aware vs Random Baseline"
)

plt.xlabel(
    "Method"
)

plt.ylabel(
    "Mean Confidence Change"
)

plt.xticks(
    rotation=20
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "mean_confidence_comparison.png",
    dpi=150
)

plt.close()


# ==========================================================
# Plot 2: Survival rate
# ==========================================================

plt.figure(
    figsize=(9, 6)
)

plt.bar(
    plot_df["method"],
    plot_df[
        "survival_rate"
    ]
)

plt.title(
    "Detection Survival Rate: "
    "Role-Aware vs Random Baseline"
)

plt.xlabel(
    "Method"
)

plt.ylabel(
    "Survival Rate (%)"
)

plt.xticks(
    rotation=20
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "survival_rate_comparison.png",
    dpi=150
)

plt.close()


# ==========================================================
# Plot 3: Mean bbox IoU
# ==========================================================

plt.figure(
    figsize=(9, 6)
)

plt.bar(
    plot_df["method"],
    plot_df[
        "mean_bbox_iou"
    ]
)

plt.title(
    "Mean Bounding Box IoU: "
    "Role-Aware vs Random Baseline"
)

plt.xlabel(
    "Method"
)

plt.ylabel(
    "Mean Bounding Box IoU"
)

plt.xticks(
    rotation=20
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "bbox_iou_comparison.png",
    dpi=150
)

plt.close()


print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"Saved to: "
    f"{OUTPUT_DIR}"
)