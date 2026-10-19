import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr


# ==========================================================
# Configuration
# ==========================================================

ROLE_RESULTS_PATH = "results/batch_results_validated.csv"

DRISE_RANKING_PATH = (
    "results/drise_role_analysis/"
    "drise_role_ranking.csv"
)

OUTPUT_DIR = Path(
    "results/drise_rank_analysis"
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
# Load
# ==========================================================

role_df = pd.read_csv(
    ROLE_RESULTS_PATH
)

drise_df = pd.read_csv(
    DRISE_RANKING_PATH
)


# ==========================================================
# Keep selected targets and mask perturbations
# ==========================================================

role_df = role_df[
    role_df["target_ann_id"].isin(
        TARGET_IDS
    )
].copy()

role_df = role_df[
    role_df["perturbation_type"]
    .astype(str)
    .str.lower()
    == "mask"
].copy()

drise_df = drise_df[
    drise_df["target_ann_id"].isin(
        TARGET_IDS
    )
].copy()


# ==========================================================
# Build comparison
# ==========================================================

rows = []

for target_id in TARGET_IDS:

    target_rows = role_df[
        role_df["target_ann_id"]
        == target_id
    ]

    drise_row = drise_df[
        drise_df["target_ann_id"]
        == target_id
    ]

    if (
        target_rows.empty
        or drise_row.empty
    ):
        continue


    drise_row = drise_row.iloc[0]


    role_scores = {}

    for role in [
        "target",
        "occluder",
        "context"
    ]:

        row = target_rows[
            target_rows["role"]
            == role
        ]

        if row.empty:
            role_scores[role] = np.nan
            continue

        row = row.iloc[0]

        original_confidence = float(
            row["original_confidence"]
        )

        confidence_change = row[
            "confidence_change"
        ]

        survived = row[
            "survived"
        ]

        if isinstance(
            survived,
            str
        ):
            survived = (
                survived.lower()
                == "true"
            )


        # If detection disappears:
        # perturbed confidence is effectively 0
        # so confidence drop = original confidence
        if not survived:

            score = (
                original_confidence
            )

        elif pd.isna(
            confidence_change
        ):

            score = (
                original_confidence
            )

        else:

            score = float(
                confidence_change
            )


        role_scores[
            role
        ] = score


    drise_scores = {
        "target": float(
            drise_row["target"]
        ),

        "occluder": float(
            drise_row["occluder"]
        ),

        "context": float(
            drise_row["context"]
        )
    }


    # ======================================================
    # Arrays in same role order
    # ======================================================

    role_order = [
        "target",
        "occluder",
        "context"
    ]


    perturbation_values = [
        role_scores[
            role
        ]
        for role in role_order
    ]

    drise_values = [
        drise_scores[
            role
        ]
        for role in role_order
    ]


    # ======================================================
    # Spearman correlation
    # ======================================================

    correlation, p_value = (
        spearmanr(
            perturbation_values,
            drise_values
        )
    )


    # ======================================================
    # Rankings
    # ======================================================

    perturbation_ranking = sorted(
        role_order,
        key=lambda role:
        role_scores[role],
        reverse=True
    )

    drise_ranking = sorted(
        role_order,
        key=lambda role:
        drise_scores[role],
        reverse=True
    )


    rows.append(
        {
            "target_ann_id":
                target_id,

            "role_target_score":
                role_scores[
                    "target"
                ],

            "role_occluder_score":
                role_scores[
                    "occluder"
                ],

            "role_context_score":
                role_scores[
                    "context"
                ],

            "drise_target":
                drise_scores[
                    "target"
                ],

            "drise_occluder":
                drise_scores[
                    "occluder"
                ],

            "drise_context":
                drise_scores[
                    "context"
                ],

            "role_ranking":
                " > ".join(
                    perturbation_ranking
                ),

            "drise_ranking":
                " > ".join(
                    drise_ranking
                ),

            "spearman_correlation":
                correlation,

            "spearman_p_value":
                p_value
        }
    )


# ==========================================================
# DataFrame
# ==========================================================

result_df = pd.DataFrame(
    rows
)


# ==========================================================
# Save
# ==========================================================

output_path = (
    OUTPUT_DIR
    / "drise_rank_correlation.csv"
)

result_df.to_csv(
    output_path,
    index=False
)


# ==========================================================
# Summary
# ==========================================================

mean_correlation = (
    result_df[
        "spearman_correlation"
    ].mean()
)

median_correlation = (
    result_df[
        "spearman_correlation"
    ].median()
)


positive_count = int(
    (
        result_df[
            "spearman_correlation"
        ] > 0
    ).sum()
)


negative_count = int(
    (
        result_df[
            "spearman_correlation"
        ] < 0
    ).sum()
)


perfect_count = int(
    (
        result_df[
            "spearman_correlation"
        ] == 1.0
    ).sum()
)


# ==========================================================
# Print
# ==========================================================

print()
print("=" * 100)
print("D-RISE VS ROLE-AWARE RANK CORRELATION")
print("=" * 100)


print(
    result_df[
        [
            "target_ann_id",
            "role_ranking",
            "drise_ranking",
            "spearman_correlation"
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)

print(
    f"Mean Spearman correlation: "
    f"{mean_correlation:.4f}"
)

print(
    f"Median Spearman correlation: "
    f"{median_correlation:.4f}"
)

print(
    f"Positive correlation targets: "
    f"{positive_count}/{len(result_df)}"
)

print(
    f"Negative correlation targets: "
    f"{negative_count}/{len(result_df)}"
)

print(
    f"Perfect ranking agreement: "
    f"{perfect_count}/{len(result_df)}"
)

print()
print(
    f"Saved to: {output_path}"
)