import pandas as pd
from pathlib import Path


# ==========================================================
# Configuration
# ==========================================================

ROLE_RESULTS_PATH = "results/batch_results_validated.csv"

DRISE_RANKING_PATH = (
    "results/drise_role_analysis/"
    "drise_role_ranking.csv"
)

OUTPUT_DIR = Path(
    "results/drise_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TARGET_IDS = [
    68,
    148,
    743,
    1470,
    1873
]


# ==========================================================
# Load data
# ==========================================================

role_df = pd.read_csv(
    ROLE_RESULTS_PATH
)

drise_df = pd.read_csv(
    DRISE_RANKING_PATH
)


print("Role-aware CSV columns:")
print(role_df.columns.tolist())
print()


# ==========================================================
# Detect column names
# ==========================================================

def find_column(df, candidates):

    for candidate in candidates:

        if candidate in df.columns:
            return candidate

    return None


target_id_col = find_column(
    role_df,
    [
        "target_ann_id",
        "target_id",
        "annotation_id"
    ]
)

role_col = find_column(
    role_df,
    [
        "role",
        "scene_role"
    ]
)

perturbation_col = find_column(
    role_df,
    [
        "perturbation",
        "perturbation_type",
        "perturb_type",
        "method"
    ]
)

confidence_change_col = find_column(
    role_df,
    [
        "confidence_change",
        "conf_change",
        "delta_confidence",
        "delta_conf"
    ]
)

survived_col = find_column(
    role_df,
    [
        "survived",
        "detection_survived"
    ]
)


# ==========================================================
# Validate required columns
# ==========================================================

required = {
    "target ID": target_id_col,
    "role": role_col,
    "confidence change": confidence_change_col
}


missing = [
    name
    for name, column in required.items()
    if column is None
]


if missing:

    raise ValueError(
        "Could not find required columns: "
        + ", ".join(missing)
        + "\nAvailable columns: "
        + str(role_df.columns.tolist())
    )


print(
    f"Target ID column: {target_id_col}"
)

print(
    f"Role column: {role_col}"
)

print(
    f"Perturbation column: {perturbation_col}"
)

print(
    f"Confidence change column: "
    f"{confidence_change_col}"
)

print(
    f"Survived column: {survived_col}"
)

print()


# ==========================================================
# Keep selected D-RISE targets
# ==========================================================

role_df = role_df[
    role_df[target_id_col].isin(
        TARGET_IDS
    )
].copy()


drise_df = drise_df[
    drise_df[
        "target_ann_id"
    ].isin(
        TARGET_IDS
    )
].copy()


# ==========================================================
# Keep MASK perturbation only
# ==========================================================

if perturbation_col is not None:

    print(
        "Perturbation values:",
        role_df[
            perturbation_col
        ].unique().tolist()
    )

    role_df = role_df[
        role_df[
            perturbation_col
        ].astype(str).str.lower()
        == "mask"
    ].copy()

else:

    print(
        "WARNING: No perturbation column found."
    )

    print(
        "Assuming the CSV already contains "
        "the required mask results."
    )


# ==========================================================
# Compare targets
# ==========================================================

comparison_rows = []


for target_id in TARGET_IDS:

    target_rows = role_df[
        role_df[
            target_id_col
        ] == target_id
    ]


    if target_rows.empty:

        print(
            f"Skipping {target_id}: "
            f"no role-aware results"
        )

        continue


    role_scores = {}


    for role in [
        "target",
        "occluder",
        "context"
    ]:

        row = target_rows[
            target_rows[
                role_col
            ] == role
        ]


        if row.empty:
            continue


        row = row.iloc[0]


        confidence_change = row[
            confidence_change_col
        ]


        # ----------------------------------------------
        # If detection disappeared, treat this as
        # maximum perturbation importance.
        # ----------------------------------------------

        if survived_col is not None:

            survived = row[
                survived_col
            ]

            if isinstance(
                survived,
                str
            ):

                survived = (
                    survived.lower()
                    == "true"
                )

            if not survived:

                score = float(
                    "inf"
                )

            elif pd.isna(
                confidence_change
            ):

                score = float(
                    "inf"
                )

            else:

                score = float(
                    confidence_change
                )

        else:

            if pd.isna(
                confidence_change
            ):

                score = float(
                    "inf"
                )

            else:

                score = float(
                    confidence_change
                )


        role_scores[
            role
        ] = score


    if not role_scores:

        print(
            f"Skipping {target_id}: "
            f"no usable role scores"
        )

        continue


    role_method_top = max(
        role_scores,
        key=role_scores.get
    )


    # ======================================================
    # D-RISE
    # ======================================================

    drise_row = drise_df[
        drise_df[
            "target_ann_id"
        ] == target_id
    ]


    if drise_row.empty:

        print(
            f"Skipping {target_id}: "
            f"no D-RISE result"
        )

        continue


    drise_row = drise_row.iloc[0]


    drise_top = drise_row[
        "highest_saliency_role"
    ]


    agreement = (
        role_method_top
        == drise_top
    )


    comparison_rows.append(
        {
            "target_ann_id":
                target_id,

            "role_target_score":
                role_scores.get(
                    "target"
                ),

            "role_occluder_score":
                role_scores.get(
                    "occluder"
                ),

            "role_context_score":
                role_scores.get(
                    "context"
                ),

            "role_method_top":
                role_method_top,

            "drise_target_saliency":
                drise_row.get(
                    "target"
                ),

            "drise_occluder_saliency":
                drise_row.get(
                    "occluder"
                ),

            "drise_context_saliency":
                drise_row.get(
                    "context"
                ),

            "drise_top":
                drise_top,

            "agreement":
                agreement
        }
    )


# ==========================================================
# Save
# ==========================================================

comparison_df = pd.DataFrame(
    comparison_rows
)


comparison_path = (
    OUTPUT_DIR
    / "drise_vs_role_comparison.csv"
)


comparison_df.to_csv(
    comparison_path,
    index=False
)

# ==========================================================
# Detailed scores
# ==========================================================

print()
print("=" * 100)
print("DETAILED SCORES")
print("=" * 100)

print(
    comparison_df[
        [
            "target_ann_id",
            "role_target_score",
            "role_occluder_score",
            "role_context_score",
            "drise_target_saliency",
            "drise_occluder_saliency",
            "drise_context_saliency",
        ]
    ].to_string(
        index=False
    )
)


# ==========================================================
# Summary
# ==========================================================

total = len(
    comparison_df
)

if total > 0:

    agreements = int(
        comparison_df[
            "agreement"
        ].sum()
    )

    agreement_rate = (
        agreements
        / total
        * 100
    )

else:

    agreements = 0
    agreement_rate = 0.0


print()
print("=" * 80)
print("D-RISE VS ROLE-AWARE COMPARISON")
print("=" * 80)


if not comparison_df.empty:

    print(
        comparison_df[
            [
                "target_ann_id",
                "role_method_top",
                "drise_top",
                "agreement"
            ]
        ].to_string(
            index=False
        )
    )


print()

print(
    f"Agreement: "
    f"{agreements}/{total}"
)

print(
    f"Agreement rate: "
    f"{agreement_rate:.2f}%"
)


print()

print(
    f"Saved comparison to: "
    f"{comparison_path}"
)