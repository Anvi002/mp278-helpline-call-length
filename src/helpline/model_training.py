
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.helpline.config import load_config
from src.helpline.mlflow_tracking import log_training_run


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

CATEGORICAL_FEATURES = [
    "language",
    "team"
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "handle_min"


def load_datasets():
    train = pd.read_csv(DATA_DIR / "train_clean.csv")
    later = pd.read_csv(DATA_DIR / "later_clean.csv")

    return train, later


def prepare_check_data(later):
    """Use the final 30% of later data for evaluation."""

    later = later.copy()
    later["date"] = pd.to_datetime(later["date"])

    later = later.sort_values("date").reset_index(drop=True)

    split_index = int(len(later) * 0.70)

    check_data = later.iloc[split_index:].copy()

    return check_data


def build_model(config):
    """Create preprocessing and Random Forest pipeline."""

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES
            )
        ]
    )

    model_config = config["model"]

    random_forest = RandomForestRegressor(
        n_estimators=model_config["n_estimators"],
        max_depth=model_config["max_depth"],
        min_samples_split=model_config["min_samples_split"],
        min_samples_leaf=model_config["min_samples_leaf"],
        random_state=config["project"]["random_state"],
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("regressor", random_forest)
    ])

    return pipeline


def train_v1():
    config = load_config()

    train, later = load_datasets()

    check = prepare_check_data(later)

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_check = check[FEATURES]
    y_check = check[TARGET]

    model = build_model(config)

    print("Training v1 model...")
    print("Training rows:", len(train))
    print("Check rows:", len(check))

    model.fit(X_train, y_train)

    predictions = model.predict(X_check)

    mae = mean_absolute_error(y_check, predictions)

    print("\nV1 MODEL RESULTS")
    print("MAE:", round(mae, 4))

    parameters = {
        "model": config["model"]["name"],
        "n_estimators": config["model"]["n_estimators"],
        "max_depth": config["model"]["max_depth"],
        "min_samples_split": config["model"]["min_samples_split"],
        "min_samples_leaf": config["model"]["min_samples_leaf"],
        "random_state": config["project"]["random_state"],
        "training_rows": len(train),
        "check_rows": len(check),
        "evaluation": "last_30_percent_of_later"
    }

    run_id = log_training_run(
        model=model,
        parameters=parameters,
        mae=mae
    )

    print("\nMLflow run ID:", run_id)

    return model, mae, run_id


if __name__ == "__main__":
    train_v1()
