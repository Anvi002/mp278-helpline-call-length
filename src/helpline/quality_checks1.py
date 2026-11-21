import pandas as pd

df = pd.read_csv("data/raw/train.csv")

print("\n========== QUEUE WAIT RANGES ==========")

df["queue_range"] = pd.cut(
    df["queue_wait_sec"],
    bins=[0, 30, 60, 120, 300, 480, 600],
    include_lowest=True
)

print(df["queue_range"].value_counts().sort_index())

print("\n========== POSSIBLE SENTINEL COUNTS ==========")

for col in df.select_dtypes(include="number").columns:
    suspicious_values = [-999, -1, 0, 99, 999, 9999]

    found = {
        value: int((df[col] == value).sum())
        for value in suspicious_values
        if (df[col] == value).any()
    }

    if found:
        print(col, found)

print("\n========== DUPLICATE RECORD IDS ==========")

print(
    df[df["record_id"].duplicated(keep=False)]
    [["record_id", "date", "issue_count", "handle_min"]]
    .sort_values("record_id")
    .to_string(index=False)
)

print("\n========== DATE-WISE QUEUE WAIT ==========")

df["date"] = pd.to_datetime(df["date"])

print(
    df.groupby(df["date"].dt.month)["queue_wait_sec"]
    .agg(["count", "mean", "median", "min", "max"])
)