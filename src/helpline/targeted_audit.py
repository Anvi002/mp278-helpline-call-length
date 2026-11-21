
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/train.csv")


def investigate():
    df = pd.read_csv(DATA_PATH)

    # ---------------------------------
    # 1. Investigate possible leakage
    # ---------------------------------
    print("\n" + "=" * 60)
    print("1. LEAKAGE INVESTIGATION")
    print("=" * 60)

    print("\nCorrelation with handle_min:")
    print(
        df.select_dtypes(include="number")
        .corr()["handle_min"]
        .sort_values(ascending=False)
    )

    print("\nCompare phone bill duration with target:")
    df["duration_difference"] = (
        df["duration_from_phone_bill"] - df["handle_min"]
    )

    print(df[
        ["duration_from_phone_bill", "handle_min",
         "duration_difference"]
    ].describe())

    print("\nSample records:")
    print(df[
        ["record_id", "duration_from_phone_bill", "handle_min"]
    ].head(15).to_string(index=False))

    # ---------------------------------
    # 2. Investigate unit mix-up
    # ---------------------------------
    print("\n" + "=" * 60)
    print("2. UNIT MIX-UP INVESTIGATION")
    print("=" * 60)

    print("\nQueue wait time summary:")
    print(df["queue_wait_sec"].describe())

    print("\nQueue wait time quantiles:")
    print(df["queue_wait_sec"].quantile(
        [0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1]
    ))

    print("\nQueue wait times below 60:")
    print(df[df["queue_wait_sec"] < 60][
        ["record_id", "queue_wait_sec", "handle_min"]
    ].to_string(index=False))

    print("\nQueue wait times above 500:")
    print(df[df["queue_wait_sec"] > 500][
        ["record_id", "queue_wait_sec", "handle_min"]
    ].to_string(index=False))

    # ---------------------------------
    # 3. Investigate unusually high target
    # ---------------------------------
    print("\n" + "=" * 60)
    print("3. UNUSUAL TARGET VALUES")
    print("=" * 60)

    print("\nHighest handle_min values:")
    print(
        df.nlargest(20, "handle_min")[
            ["record_id", "date", "queue_wait_sec",
             "duration_from_phone_bill", "handle_min"]
        ].to_string(index=False)
    )

    # ---------------------------------
    # 4. Investigate duplicate records
    # ---------------------------------
    print("\n" + "=" * 60)
    print("4. DUPLICATE INVESTIGATION")
    print("=" * 60)

    duplicate_rows = df[df.duplicated(keep=False)]

    print("Rows involved in exact duplicates:",
          len(duplicate_rows))

    print("\nSample duplicate records:")
    print(duplicate_rows.head(10).to_string(index=False))


if __name__ == "__main__":
    investigate()
