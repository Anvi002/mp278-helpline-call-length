
import pandas as pd

df = pd.read_csv("data/raw/train.csv")

# Sort by date and record ID to inspect consecutive records
df = df.sort_values(["date", "record_id"]).reset_index(drop=True)

print("\n========== CONSECUTIVE REPEATED VALUES ==========")

columns = [
    "queue_wait_sec",
    "agent_exp_months",
    "previous_calls",
    "issue_count",
    "hour",
    "caller_age",
    "duration_from_phone_bill"
]

for col in columns:
    groups = df[col].ne(df[col].shift()).cumsum()
    run_lengths = df.groupby(groups).size()
    longest = run_lengths.max()

    print(f"\n{col}: longest consecutive repetition = {longest}")

    if longest >= 5:
        longest_group = run_lengths.idxmax()
        print(df.loc[
            groups == longest_group,
            ["record_id", "date", col]
        ].to_string(index=False))

# Investigate target anomalies
print("\n========== TARGET ANOMALY INVESTIGATION ==========")

df["target_ratio"] = (
    df["handle_min"] / df["duration_from_phone_bill"]
)

suspicious = df[df["handle_min"] > 30]

print(
    suspicious[
        ["record_id", "duration_from_phone_bill",
         "handle_min", "target_ratio"]
    ].to_string(index=False)
)

# Compare target and phone-bill duration relationship
print("\nCorrelation:")
print(
    df["duration_from_phone_bill"].corr(df["handle_min"])
)

print("\nTarget difference statistics:")
print(
    (df["handle_min"] - df["duration_from_phone_bill"]).describe()
)
print("\n========== TARGET TYPO PATTERN ==========")

suspicious = df[df["handle_min"] > 30].copy()

suspicious["corrected_candidate"] = (
    suspicious["handle_min"] / 10
)

print(
    suspicious[
        [
            "record_id",
            "handle_min",
            "corrected_candidate",
            "duration_from_phone_bill"
        ]
    ].to_string(index=False)
)

print("\nDifference after dividing target by 10:")

print(
    (
        suspicious["corrected_candidate"]
        - suspicious["duration_from_phone_bill"]
    ).describe()
)