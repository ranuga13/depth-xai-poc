import numpy as np
import pandas as pd
from pathlib import Path


# ==========================================================
# Configuration
# ==========================================================

KINS_RESULTS_PATH = Path(
    "results/batch_results_validated.csv"
)

AUTO_RESULTS_PATH = Path(
    "results/batch_results_auto_roles.csv"
)

AUTO_VALIDATION_PATH = Path(
    "results/automatic_role_validation/automatic_role_validation.csv"
)

OUTPUT_DIR = Path(
    "results/auto_vs_kins_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# Load data
# ==========================================================

kins_df = pd.read_csv(
    KINS_RESULTS_PATH
)

auto_df = pd.read_csv(
    AUTO_RESULTS_PATH
)

validation_df = pd.read_csv(
    AUTO_VALIDATION_PATH
)


print("=" * 90)
print("AUTO ROLE VS KINS ROLE COMPARISON")
print("=" * 90)

print(
    f"KINS rows: {len(kins_df)}"
)

print(
    f"Auto rows: {len(auto_df)}"
)

print(
    f"KINS targets: "
    f"{kins_df['target_ann_id'].nunique()}"
)

print(
    f"Auto targets: "
    f"{auto_df['target_ann_id'].nunique()}"
)


# ==========================================================
# Ensure survival is boolean
# ==========================================================

def normalize_survived(series):

    if series.dtype == bool:
        return series

    return (
        series.astype(str)
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False
            }
        )
    )


kins_df["survived"] = normalize_survived(
    kins_df["survived"]
)

auto_df["survived"] = normalize_survived(
    auto_df["survived"]
)


# ==========================================================
# Summary function
# ==========================================================

def summarize_results(
    df,
    method_name
):

    grouped = (
        df.groupby(
            [
                "role",
                "perturbation_type"
            ]
        )
        .agg(
            samples=(
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


    grouped[
        "survival_rate"
    ] *= 100


    grouped.insert(
        0,
        "method",
        method_name
    )


    return grouped


# ==========================================================
# Overall summaries
# ==========================================================

kins_summary = summarize_results(
    kins_df,
    "KINS"
)

auto_summary = summarize_results(
    auto_df,
    "AUTO"
)


summary_df = pd.concat(
    [
        kins_summary,
        auto_summary
    ],
    ignore_index=True
)


summary_path = (
    OUTPUT_DIR
    / "summary_comparison.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


print()
print("=" * 90)
print("AGGREGATE SUMMARY")
print("=" * 90)

print(
    summary_df.to_string(
        index=False
    )
)


# ==========================================================
# Merge paired results
# ==========================================================

merge_columns = [
    "target_ann_id",
    "role",
    "perturbation_type"
]


kins_keep = kins_df[
    merge_columns
    + [
        "confidence_change",
        "bbox_iou",
        "survived"
    ]
].copy()


auto_keep = auto_df[
    merge_columns
    + [
        "confidence_change",
        "bbox_iou",
        "survived"
    ]
].copy()


paired_df = pd.merge(
    kins_keep,
    auto_keep,
    on=merge_columns,
    how="inner",
    suffixes=(
        "_kins",
        "_auto"
    )
)


# ==========================================================
# Difference metrics
# ==========================================================

paired_df[
    "confidence_difference"
] = (
    paired_df[
        "confidence_change_auto"
    ]
    -
    paired_df[
        "confidence_change_kins"
    ]
)


paired_df[
    "abs_confidence_difference"
] = (
    paired_df[
        "confidence_difference"
    ].abs()
)


paired_df[
    "bbox_iou_difference"
] = (
    paired_df[
        "bbox_iou_auto"
    ]
    -
    paired_df[
        "bbox_iou_kins"
    ]
)


paired_df[
    "survival_agreement"
] = (
    paired_df[
        "survived_auto"
    ]
    ==
    paired_df[
        "survived_kins"
    ]
)


# ==========================================================
# Add role-inference correctness
# ==========================================================

validation_keep = validation_df[
    [
        "target_ann_id",
        "status"
    ]
].copy()


paired_df = pd.merge(
    paired_df,
    validation_keep,
    on="target_ann_id",
    how="left"
)


paired_path = (
    OUTPUT_DIR
    / "paired_results.csv"
)

paired_df.to_csv(
    paired_path,
    index=False
)


# ==========================================================
# Pairwise comparison by role / perturbation
# ==========================================================

comparison_rows = []


for (
    role,
    perturbation
), group in paired_df.groupby(
    [
        "role",
        "perturbation_type"
    ]
):

    valid_conf = group.dropna(
        subset=[
            "confidence_change_kins",
            "confidence_change_auto"
        ]
    )


    valid_iou = group.dropna(
        subset=[
            "bbox_iou_kins",
            "bbox_iou_auto"
        ]
    )


    mean_abs_conf_diff = (
        valid_conf[
            "abs_confidence_difference"
        ].mean()
        if len(valid_conf) > 0
        else np.nan
    )


    median_abs_conf_diff = (
        valid_conf[
            "abs_confidence_difference"
        ].median()
        if len(valid_conf) > 0
        else np.nan
    )


    mean_iou_diff = (
        valid_iou[
            "bbox_iou_difference"
        ].mean()
        if len(valid_iou) > 0
        else np.nan
    )


    survival_agreement = (
        group[
            "survival_agreement"
        ].mean()
        * 100
    )


    comparison_rows.append(
        {
            "role":
                role,

            "perturbation_type":
                perturbation,

            "samples":
                len(group),

            "paired_confidence_samples":
                len(valid_conf),

            "mean_abs_confidence_difference":
                mean_abs_conf_diff,

            "median_abs_confidence_difference":
                median_abs_conf_diff,

            "mean_bbox_iou_difference":
                mean_iou_diff,

            "survival_agreement_percent":
                survival_agreement
        }
    )


comparison_df = pd.DataFrame(
    comparison_rows
)


comparison_path = (
    OUTPUT_DIR
    / "paired_comparison_summary.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False
)


print()
print("=" * 90)
print("PAIRED KINS VS AUTO COMPARISON")
print("=" * 90)

print(
    comparison_df.to_string(
        index=False
    )
)


# ==========================================================
# Separate correctly inferred role cases
# ==========================================================

correct_role_df = paired_df[
    paired_df[
        "status"
    ]
    == "correct"
].copy()


wrong_role_df = paired_df[
    paired_df[
        "status"
    ]
    != "correct"
].copy()


print()
print("=" * 90)
print("ROLE INFERENCE SPLIT")
print("=" * 90)

print(
    f"Rows with correct role inference: "
    f"{len(correct_role_df)}"
)

print(
    f"Rows with incorrect role inference: "
    f"{len(wrong_role_df)}"
)

print(
    f"Correct-role targets: "
    f"{correct_role_df['target_ann_id'].nunique()}"
)

print(
    f"Incorrect-role targets: "
    f"{wrong_role_df['target_ann_id'].nunique()}"
)


# ==========================================================
# Analyze only the 38 correctly inferred targets
# ==========================================================

correct_comparison_rows = []


for (
    role,
    perturbation
), group in correct_role_df.groupby(
    [
        "role",
        "perturbation_type"
    ]
):

    valid_conf = group.dropna(
        subset=[
            "confidence_change_kins",
            "confidence_change_auto"
        ]
    )


    valid_iou = group.dropna(
        subset=[
            "bbox_iou_kins",
            "bbox_iou_auto"
        ]
    )


    correct_comparison_rows.append(
        {
            "role":
                role,

            "perturbation_type":
                perturbation,

            "samples":
                len(group),

            "paired_confidence_samples":
                len(valid_conf),

            "mean_abs_confidence_difference":
                (
                    valid_conf[
                        "abs_confidence_difference"
                    ].mean()
                    if len(valid_conf) > 0
                    else np.nan
                ),

            "median_abs_confidence_difference":
                (
                    valid_conf[
                        "abs_confidence_difference"
                    ].median()
                    if len(valid_conf) > 0
                    else np.nan
                ),

            "mean_bbox_iou_difference":
                (
                    valid_iou[
                        "bbox_iou_difference"
                    ].mean()
                    if len(valid_iou) > 0
                    else np.nan
                ),

            "survival_agreement_percent":
                (
                    group[
                        "survival_agreement"
                    ].mean()
                    * 100
                )
        }
    )


correct_comparison_df = pd.DataFrame(
    correct_comparison_rows
)


correct_path = (
    OUTPUT_DIR
    / "correct_role_only_comparison.csv"
)

correct_comparison_df.to_csv(
    correct_path,
    index=False
)


print()
print("=" * 90)
print("COMPARISON ON 38 CORRECTLY INFERRED TARGETS")
print("=" * 90)

print(
    correct_comparison_df.to_string(
        index=False
    )
)


# ==========================================================
# Analyze the two incorrect role cases
# ==========================================================

if not wrong_role_df.empty:

    wrong_targets = sorted(
        wrong_role_df[
            "target_ann_id"
        ].unique()
    )

    print()
    print("=" * 90)
    print("INCORRECT ROLE INFERENCE TARGETS")
    print("=" * 90)

    print(
        f"Targets: {wrong_targets}"
    )


    wrong_case_summary = (
        wrong_role_df[
            [
                "target_ann_id",
                "role",
                "perturbation_type",
                "confidence_change_kins",
                "confidence_change_auto",
                "confidence_difference",
                "bbox_iou_kins",
                "bbox_iou_auto",
                "survived_kins",
                "survived_auto"
            ]
        ]
        .sort_values(
            [
                "target_ann_id",
                "role",
                "perturbation_type"
            ]
        )
    )


    print(
        wrong_case_summary.to_string(
            index=False
        )
    )


    wrong_path = (
        OUTPUT_DIR
        / "incorrect_role_cases.csv"
    )

    wrong_case_summary.to_csv(
        wrong_path,
        index=False
    )


# ==========================================================
# Overall agreement
# ==========================================================

overall_survival_agreement = (
    paired_df[
        "survival_agreement"
    ].mean()
    * 100
)


valid_conf_all = paired_df.dropna(
    subset=[
        "confidence_change_kins",
        "confidence_change_auto"
    ]
)


mean_abs_difference_all = (
    valid_conf_all[
        "abs_confidence_difference"
    ].mean()
)


median_abs_difference_all = (
    valid_conf_all[
        "abs_confidence_difference"
    ].median()
)


print()
print("=" * 90)
print("OVERALL AGREEMENT")
print("=" * 90)

print(
    f"Paired rows: "
    f"{len(paired_df)}"
)

print(
    f"Paired numeric confidence rows: "
    f"{len(valid_conf_all)}"
)

print(
    f"Mean absolute confidence-change difference: "
    f"{mean_abs_difference_all:.6f}"
)

print(
    f"Median absolute confidence-change difference: "
    f"{median_abs_difference_all:.6f}"
)

print(
    f"Survival agreement: "
    f"{overall_survival_agreement:.2f}%"
)


print()
print("=" * 90)
print("FILES SAVED")
print("=" * 90)

print(
    summary_path
)

print(
    paired_path
)

print(
    comparison_path
)

print(
    correct_path
)

print()
print("Comparison complete.")