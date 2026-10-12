import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ==========================================================
# Configuration
# ==========================================================

ROLE_RESULTS_CSV = "results/batch_results_validated.csv"
BASELINE_RESULTS_CSV = "results/random_baseline_results.csv"

OUTPUT_DIR = Path(
    "results/paired_baseline_analysis"
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


# ==========================================================
# Keep only mask results from role-aware method
# ==========================================================

role_mask_df = role_df[
    role_df["perturbation_type"] == "mask"
].copy()


# ==========================================================
# Average random baseline per target
# ==========================================================

baseline_per_target = (
    baseline_df
    .groupby("target_ann_id")
    .agg(
        random_mean_conf_change=(
            "confidence_change",
            "mean"
        ),

        random_median_conf_change=(
            "confidence_change",
            "median"
        ),

        random_survival_rate=(
            "survived",
            "mean"
        ),

        random_mean_bbox_iou=(
            "bbox_iou",
            "mean"
        )
    )
    .reset_index()
)


baseline_per_target[
    "random_survival_rate"
] *= 100


# ==========================================================
# Get role-aware mask results per target
# ==========================================================

role_conf = role_mask_df.pivot_table(
    index="target_ann_id",
    columns="role",
    values="confidence_change"
).reset_index()


role_conf = role_conf.rename(
    columns={
        "target": "target_conf_change",
        "context": "context_conf_change",
        "occluder": "occluder_conf_change"
    }
)


# ==========================================================
# Merge
# ==========================================================

paired = pd.merge(
    baseline_per_target,
    role_conf,
    on="target_ann_id",
    how="inner"
)


# ==========================================================
# Calculate paired differences
# ==========================================================

paired[
    "target_minus_random"
] = (
    paired["target_conf_change"]
    - paired["random_mean_conf_change"]
)

paired[
    "context_minus_random"
] = (
    paired["context_conf_change"]
    - paired["random_mean_conf_change"]
)

paired[
    "occluder_minus_random"
] = (
    paired["occluder_conf_change"]
    - paired["random_mean_conf_change"]
)


# ==========================================================
# Count stronger-than-random cases
# ==========================================================

target_valid = paired[
    "target_minus_random"
].dropna()

context_valid = paired[
    "context_minus_random"
].dropna()

occluder_valid = paired[
    "occluder_minus_random"
].dropna()


target_stronger = (
    target_valid > 0
).sum()

context_stronger = (
    context_valid > 0
).sum()

occluder_stronger = (
    occluder_valid > 0
).sum()


# ==========================================================
# Summary
# ==========================================================

summary = pd.DataFrame(
    [
        {
            "role": "target",
            "num_valid_targets": len(
                target_valid
            ),
            "stronger_than_random_count": (
                target_stronger
            ),
            "stronger_than_random_percent": (
                target_stronger
                / len(target_valid)
                * 100
                if len(target_valid) > 0
                else 0
            ),
            "mean_paired_difference": (
                target_valid.mean()
            ),
            "median_paired_difference": (
                target_valid.median()
            )
        },

        {
            "role": "context",
            "num_valid_targets": len(
                context_valid
            ),
            "stronger_than_random_count": (
                context_stronger
            ),
            "stronger_than_random_percent": (
                context_stronger
                / len(context_valid)
                * 100
                if len(context_valid) > 0
                else 0
            ),
            "mean_paired_difference": (
                context_valid.mean()
            ),
            "median_paired_difference": (
                context_valid.median()
            )
        },

        {
            "role": "occluder",
            "num_valid_targets": len(
                occluder_valid
            ),
            "stronger_than_random_count": (
                occluder_stronger
            ),
            "stronger_than_random_percent": (
                occluder_stronger
                / len(occluder_valid)
                * 100
                if len(occluder_valid) > 0
                else 0
            ),
            "mean_paired_difference": (
                occluder_valid.mean()
            ),
            "median_paired_difference": (
                occluder_valid.median()
            )
        }
    ]
)


# ==========================================================
# Print results
# ==========================================================

print("=" * 70)
print("PAIRED ROLE-AWARE VS RANDOM BASELINE")
print("=" * 70)

print(
    f"Targets available for paired analysis: "
    f"{paired['target_ann_id'].nunique()}"
)

print()

print(
    summary.to_string(
        index=False
    )
)


# ==========================================================
# Save CSV files
# ==========================================================

paired.to_csv(
    OUTPUT_DIR
    / "paired_per_target_results.csv",
    index=False
)

summary.to_csv(
    OUTPUT_DIR
    / "paired_summary.csv",
    index=False
)


# ==========================================================
# Plot: paired differences by target
# ==========================================================

plot_df = paired[
    [
        "target_ann_id",
        "target_minus_random",
        "context_minus_random",
        "occluder_minus_random"
    ]
].set_index(
    "target_ann_id"
)


ax = plot_df.plot(
    kind="bar",
    figsize=(16, 7)
)

ax.axhline(
    y=0,
    linewidth=1
)

ax.set_title(
    "Role-Aware Confidence Change "
    "Relative to Random Baseline"
)

ax.set_xlabel(
    "Target Annotation ID"
)

ax.set_ylabel(
    "Role ΔConfidence - Random Mean ΔConfidence"
)

plt.xticks(
    rotation=90
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "paired_difference_by_target.png",
    dpi=150
)

plt.close()


# ==========================================================
# Plot: stronger-than-random percentage
# ==========================================================

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    summary["role"],
    summary[
        "stronger_than_random_percent"
    ]
)

plt.title(
    "Percentage of Targets Where "
    "Role-Aware Masking Is Stronger Than Random"
)

plt.xlabel(
    "Scene Role"
)

plt.ylabel(
    "Targets (%)"
)

plt.ylim(
    0,
    100
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "stronger_than_random_percentage.png",
    dpi=150
)

plt.close()


# ==========================================================
# Plot: average paired difference
# ==========================================================

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    summary["role"],
    summary[
        "mean_paired_difference"
    ]
)

plt.axhline(
    y=0,
    linewidth=1
)

plt.title(
    "Mean Paired Difference vs Random Baseline"
)

plt.xlabel(
    "Scene Role"
)

plt.ylabel(
    "Mean ΔConfidence Difference"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "mean_paired_difference.png",
    dpi=150
)

plt.close()


print()
print("=" * 70)
print("PAIRED ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"Saved to: {OUTPUT_DIR}"
)