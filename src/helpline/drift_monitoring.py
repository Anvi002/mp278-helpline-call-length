
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data/processed"

NUMERIC_FEATURES = [
    "queue_wait_sec",
    "agent_exp_months",
    "previous_calls",
    "issue_count",
    "hour",
    "caller_age"
]


def calculate_psi(expected, actual, bins=10):
    """Calculate PSI using training-data quantile bins."""

    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    # Create bin boundaries using training data
    breakpoints = np.unique(
        np.quantile(expected, np.linspace(0, 1, bins + 1))
    )

    # Handle features with too few unique values
    if len(breakpoints) < 2:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    # Include values outside training boundaries in edge bins
    actual_counts[0] += np.sum(actual < breakpoints[0])
    actual_counts[-1] += np.sum(actual > breakpoints[-1])

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Avoid division by zero
    epsilon = 0.0001
    expected_pct = np.clip(expected_pct, epsilon, None)
    actual_pct = np.clip(actual_pct, epsilon, None)

    psi = np.sum(
        (actual_pct - expected_pct)
        * np.log(actual_pct / expected_pct)
    )

    return float(psi)


def monitor_drift():
    train = pd.read_csv(DATA_DIR / "train_clean.csv")
    later = pd.read_csv(DATA_DIR / "later_clean.csv")

    print("\n========== POPULATION STABILITY INDEX ==========")

    psi_results = {}

    for feature in NUMERIC_FEATURES:
        psi = calculate_psi(
            train[feature],
            later[feature],
            bins=10
        )

        psi_results[feature] = psi
        print(f"{feature}: {psi:.4f}")

    most_drifted = max(psi_results, key=psi_results.get)

    print("\nMost drifted feature:", most_drifted)
    print("Highest PSI:", round(psi_results[most_drifted], 4))

    return psi_results


if __name__ == "__main__":
    monitor_drift()
