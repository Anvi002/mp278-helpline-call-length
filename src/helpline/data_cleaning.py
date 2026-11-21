
from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data/raw/train.csv"
OUTPUT_DIR = PROJECT_ROOT / "data/processed"

CLEAN_TRAIN_PATH = OUTPUT_DIR / "train_clean.csv"


def load_data(file_path):
    """Load the original dataset."""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    return pd.read_csv(file_path)


def remove_duplicates(df):
    """Remove exact duplicate rows."""
    before = len(df)

    df = df.drop_duplicates().copy()

    removed = before - len(df)

    print(f"Duplicate rows removed: {removed}")

    return df



def correct_target_typos(df):
    """Correct only the identified target decimal-point errors."""

    suspicious_ids = [
        "R00005",
        "R00311",
        "R00338",
        "R00421",
        "R00483",
        "R00514",
        "R00547",
        "R00588",
        "R00677",
        "R00708",
        "R00795",
        "R00853",
        "R00910",
        "R00986"
    ]

    mask = df["record_id"].isin(suspicious_ids)

    count = int(mask.sum())

    df.loc[mask, "handle_min"] = (
        df.loc[mask, "handle_min"] / 10
    )

    print(f"Suspicious target values corrected: {count}")

    return df



def remove_leaking_column(df):
    """Remove the target-leakage column."""
    leaking_column = "duration_from_phone_bill"

    if leaking_column in df.columns:
        df = df.drop(columns=[leaking_column])
        print(f"Removed leakage column: {leaking_column}")

    return df


def clean_training_data():
    """Run the complete training-data cleaning pipeline."""
    print("Loading original training data...")

    df = load_data(TRAIN_PATH)

    print("Original rows:", len(df))

    df = remove_duplicates(df)

    df = correct_target_typos(df)

    df = remove_leaking_column(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(CLEAN_TRAIN_PATH, index=False)

    print("\nCleaning completed!")
    print("Cleaned rows:", len(df))
    print("Cleaned columns:", len(df.columns))
    print("Saved to:", CLEAN_TRAIN_PATH)

def clean_later_data():
    """Clean later data without changing its target values."""

    later_path = PROJECT_ROOT / "data/raw/later.csv"
    clean_later_path = OUTPUT_DIR / "later_clean.csv"

    print("\nLoading later dataset...")

    df = load_data(later_path)

    print("Original rows:", len(df))

    # Remove exact duplicates
    df = remove_duplicates(df)

    # Remove target leakage column
    df = remove_leaking_column(df)

    # Do NOT correct target values in later data
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(clean_later_path, index=False)

    print("\nLater dataset cleaning completed!")
    print("Cleaned rows:", len(df))
    print("Cleaned columns:", len(df.columns))
    print("Saved to:", clean_later_path)

def clean_holdout_data():
    """Prepare holdout features for final prediction."""

    holdout_path = PROJECT_ROOT / "data/raw/holdout_features.csv"
    clean_holdout_path = OUTPUT_DIR / "holdout_clean.csv"

    print("\nLoading holdout dataset...")

    df = load_data(holdout_path)

    print("Original rows:", len(df))

    # Remove leakage column if present
    df = remove_leaking_column(df)

    # Required columns for prediction
    required_columns = [
        "record_id",
        "queue_wait_sec",
        "agent_exp_months",
        "previous_calls",
        "issue_count",
        "hour",
        "caller_age",
        "language",
        "team"
    ]

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Keep only the required columns in the correct order
    df = df[required_columns].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(clean_holdout_path, index=False)

    print("\nHoldout preparation completed!")
    print("Cleaned rows:", len(df))
    print("Cleaned columns:", len(df.columns))
    print("Saved to:", clean_holdout_path)


if __name__ == "__main__":
    clean_training_data()
    clean_later_data()
    clean_holdout_data()
