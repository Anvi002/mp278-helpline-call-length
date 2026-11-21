
import pandas as pd
from pathlib import Path

# Original training dataset
DATA_PATH = Path("data/raw/train.csv")


def inspect_dataset():

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    # Read original dataset without modifying it
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("1. DATASET OVERVIEW")
    print("=" * 60)

    print("Shape:", df.shape)
    print("Number of rows:", len(df))
    print("Number of columns:", len(df.columns))

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\n" + "=" * 60)
    print("2. DATA TYPES")
    print("=" * 60)

    print(df.dtypes)

    print("\n" + "=" * 60)
    print("3. MISSING VALUES")
    print("=" * 60)

    print(df.isnull().sum())

    print("\n" + "=" * 60)
    print("4. DUPLICATE ROWS")
    print("=" * 60)

    print("Exact duplicate rows:", df.duplicated().sum())

    print("\n" + "=" * 60)
    print("5. NUMERICAL COLUMN SUMMARY")
    print("=" * 60)

    print(df.describe(include="number").T)

    print("\n" + "=" * 60)
    print("6. CATEGORICAL COLUMN VALUES")
    print("=" * 60)

    for column in df.select_dtypes(include=["object"]).columns:
        print(f"\n{column}")
        print(df[column].value_counts(dropna=False).head(30))

    print("\n" + "=" * 60)
    print("7. FIRST 10 ROWS")
    print("=" * 60)

    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    inspect_dataset()
