
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


def evaluate_by_team():
    config = load_config()

    train, later = load_datasets()
    check = prepare_check_data(later)

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_check = check[FEATURES]
    y_check = check[TARGET]

    # Train the same v1 model
    model = build_model(config)
    model.fit(X_train, y_train)

    # Generate predictions
    predictions = model.predict(X_check)

    results = check[["team"]].copy()
    results["actual"] = y_check.values
    results["predicted"] = predictions

    print("\n========== V1 MAE BY TEAM ==========")

    for team, group in results.groupby("team"):
        mae = mean_absolute_error(
            group["actual"],
            group["predicted"]
        )

        print(f"\n{team}")
        print("Records:", len(group))
        print("MAE:", round(mae, 4), "minutes")


if __name__ == "__main__":
    evaluate_by_team()
