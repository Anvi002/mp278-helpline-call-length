
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data/processed"

# Verify training and later datasets
files = {
    "Training": DATA_DIR / "train_clean.csv",
    "Later": DATA_DIR / "later_clean.csv"
}

for name, path in files.items():
    print(f"\n========== {name} DATASET ==========")

    df = pd.read_csv(path)

    print("Shape:", df.shape)
    print("Exact duplicate rows:", df.duplicated().sum())
    print("Total missing values:", df.isnull().sum().sum())
    print("Leakage column present:",
          "duration_from_phone_bill" in df.columns)
    print("Columns:", df.columns.tolist())

train = pd.read_csv(files["Training"])
later = pd.read_csv(files["Later"])

print("\n========== COLUMN COMPARISON ==========")
print("Same columns:", set(train.columns) == set(later.columns))
print("Training-only columns:", set(train.columns) - set(later.columns))
print("Later-only columns:", set(later.columns) - set(train.columns))


# Verify holdout dataset
print("\n========== HOLDOUT DATASET ==========")

holdout_path = DATA_DIR / "holdout_clean.csv"
holdout = pd.read_csv(holdout_path)

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

print("Shape:", holdout.shape)
print("Duplicate rows:", holdout.duplicated().sum())
print("Missing values:", holdout.isnull().sum().sum())
print("Has target column:", "handle_min" in holdout.columns)
print("Has leakage column:",
      "duration_from_phone_bill" in holdout.columns)
print("Correct columns:", holdout.columns.tolist() == required_columns)

print("\nVerification completed.")
