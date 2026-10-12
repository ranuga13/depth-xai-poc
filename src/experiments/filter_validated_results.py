import pandas as pd
from pathlib import Path


INPUT_CSV = "results/batch_results.csv"
OUTPUT_CSV = "results/batch_results_validated.csv"


# Targets that failed manual validation
INVALID_TARGET_IDS = [
    4492,
    7367
]


df = pd.read_csv(
    INPUT_CSV
)


validated_df = df[
    ~df["target_ann_id"].isin(
        INVALID_TARGET_IDS
    )
].copy()


Path("results").mkdir(
    exist_ok=True
)


validated_df.to_csv(
    OUTPUT_CSV,
    index=False
)


print("=" * 60)
print("VALIDATED DATASET CREATED")
print("=" * 60)

print(
    f"Original targets: "
    f"{df['target_ann_id'].nunique()}"
)

print(
    f"Removed targets: "
    f"{INVALID_TARGET_IDS}"
)

print(
    f"Validated targets: "
    f"{validated_df['target_ann_id'].nunique()}"
)

print(
    f"Validated result rows: "
    f"{len(validated_df)}"
)

print(
    f"Saved to: {OUTPUT_CSV}"
)