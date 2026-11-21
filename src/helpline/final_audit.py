
import pandas as pd

df = pd.read_csv("data/raw/train.csv")

print("\n========== RECORD ID CHECK ==========")

print("Total rows:", len(df))
print("Unique record IDs:", df["record_id"].nunique())
print("Duplicate record IDs:", df["record_id"].duplicated().sum())

print("\n========== CATEGORICAL VALUE CHECK ==========")

for col in ["language", "team"]:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False))

print("\n========== DATE AND RECORD ID CHECK ==========")

df["date"] = pd.to_datetime(df["date"])

print("Date range:", df["date"].min(), "to", df["date"].max())

print("\nRecords per day:")
print(df.groupby("date").size().describe())

print("\n========== TARGET VALUES ABOVE 30 ==========")

print(
    df.loc[df["handle_min"] > 30,
           ["record_id", "date", "handle_min"]]
    .to_string(index=False)
)

print("\n========== NUMERIC VALUE BOUNDARIES ==========")

for col in [
    "queue_wait_sec",
    "agent_exp_months",
    "previous_calls",
    "issue_count",
    "hour",
    "caller_age"
]:
    print(
        col,
        "min =", df[col].min(),
        "max =", df[col].max(),
        "unique =", df[col].nunique()
    )
