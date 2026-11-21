
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.helpline.config import load_config
from src.helpline.model_training import (
    load_datasets,
    prepare_check_data,
    build_model,
    FEATURES,
    TARGET
)
from src.helpline.mlflow_tracking import log_training_run


def train_v2():
    config = load_config()

    train, later = load_datasets()

    # Sort later data chronologically
    later["date"] = pd.to_datetime(later["date"])
    later = later.sort_values("date").reset_index(drop=True)

    # Split later data: first 70% for training, final 30% for checking
    split_index = int(len(later) * 0.70)

    later_train = later.iloc[:split_index].copy()
    check = later.iloc[split_index:].copy()

    # Combine original training data with the first 70% of later data
    combined_train = pd.concat(
        [train, later_train],
        ignore_index=True
    )

    X_train = combined_train[FEATURES]
    y_train = combined_train[TARGET]

    X_check = check[FEATURES]
    y_check = check[TARGET]

    # Build and train v2
    model = build_model(config)

    print("Training v2 model...")
    print("Original training rows:", len(train))
    print("Additional later rows:", len(later_train))
    print("Total training rows:", len(combined_train))
    print("Check rows:", len(check))

    model.fit(X_train, y_train)

    predictions = model.predict(X_check)
    mae = mean_absolute_error(y_check, predictions)

    print("\nV2 MODEL RESULTS")
    print("MAE:", round(mae, 4))

    parameters = {
        "model": config["model"]["name"],
        "version": "v2",
        "n_estimators": config["model"]["n_estimators"],
        "max_depth": config["model"]["max_depth"],
        "min_samples_split": config["model"]["min_samples_split"],
        "min_samples_leaf": config["model"]["min_samples_leaf"],
        "random_state": config["project"]["random_state"],
        "original_training_rows": len(train),
        "additional_later_rows": len(later_train),
        "total_training_rows": len(combined_train),
        "check_rows": len(check),
        "evaluation": "same_last_30_percent_of_later"
    }

    run_id = log_training_run(
        model=model,
        parameters=parameters,
        mae=mae
    )

    print("\nMLflow run ID:", run_id)

    return model, mae, run_id


if __name__ == "__main__":
    train_v2()
