
from pathlib import Path

import pandas as pd

from src.helpline.model_registry import load_champion

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data/processed"
OUTPUT_DIR = PROJECT_ROOT / "predictions"

FEATURES = [
    "queue_wait_sec",
    "agent_exp_months",
    "previous_calls",
    "issue_count",
    "hour",
    "caller_age",
    "language",
    "team"
]


def generate_predictions():
    holdout = pd.read_csv(DATA_DIR / "holdout_clean.csv")

    # Load the registered champion model
    model = load_champion()

    # Generate predictions
    predictions = model.predict(holdout[FEATURES])

    # Create the required output
    output = pd.DataFrame({
        "record_id": holdout["record_id"],
        "prediction": predictions
    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "holdout_predictions.csv"
    output.to_csv(output_path, index=False)

    print("\n========== HOLDOUT PREDICTIONS ==========")
    print("Total predictions:", len(output))
    print("Output columns:", output.columns.tolist())
    print("Saved to:", output_path)
    print("\nFirst 5 predictions:")
    print(output.head())

    # Verify output format
    assert len(output) == 250
    assert output.columns.tolist() == ["record_id", "prediction"]
    assert output["prediction"].notna().all()

    print("\nPrediction file verification successful!")


if __name__ == "__main__":
    generate_predictions()
