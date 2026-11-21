
import hashlib
from pathlib import Path

import mlflow
import mlflow.sklearn


TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "MP278-Helpline-Call-Length"
MODEL_NAME = "mp278-helpline-call-length"

DATA_PATH = Path("data/raw/train.csv")


def setup_mlflow():
    """Configure MLflow tracking."""

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("MLflow tracking configured!")
    print("Tracking URI:", mlflow.get_tracking_uri())
    print("Experiment:", EXPERIMENT_NAME)


def calculate_data_hash():
    """Calculate SHA256 of the original training dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    sha256 = hashlib.sha256()

    with open(DATA_PATH, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def log_training_run(model, parameters, mae):
    """Log a real model training run in MLflow."""

    setup_mlflow()

    data_hash = calculate_data_hash()

    with mlflow.start_run() as run:

        # Log model configuration
        mlflow.log_params(parameters)

        # Log original dataset fingerprint
        mlflow.log_param("data_sha256", data_hash)

        # Log evaluation metric
        mlflow.log_metric("mae", float(mae))

        # Log trained model artifact
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="cloudpickle"
        )

        run_id = run.info.run_id

        print("Training run logged successfully!")
        print("Run ID:", run_id)
        print("MAE:", mae)
        print("Data SHA256:", data_hash)
        print("Model URI:", model_info.model_uri)

        return run_id
