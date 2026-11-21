
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


# MLflow configuration
TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "mp278-helpline-call-length"


def get_client():
    """Create an MLflow client."""
    mlflow.set_tracking_uri(TRACKING_URI)
    return MlflowClient(tracking_uri=TRACKING_URI)


def register_model(model_uri):
    """
    Register a trained model in MLflow Model Registry.

    model_uri: URI of an already logged model,
               for example runs:/RUN_ID/model
    """

    client = get_client()

    # Create registered model if it does not exist
    try:
        client.get_registered_model(MODEL_NAME)

    except MlflowException:
        client.create_registered_model(MODEL_NAME)

    # Register a new version
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    print("Model registered successfully!")
    print("Model name:", MODEL_NAME)
    print("Model version:", model_version.version)

    return model_version.version


def set_champion(version):
    """Assign the champion alias to a registered model version."""

    client = get_client()

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=str(version)
    )

    print(f"Version {version} assigned as champion!")


def load_champion():
    """Load the model currently assigned to the champion alias."""

    mlflow.set_tracking_uri(TRACKING_URI)

    model_uri = f"models:/{MODEL_NAME}@champion"

    model = mlflow.pyfunc.load_model(model_uri)

    print("Champion model loaded successfully!")

    return model


def list_model_versions():
    """Display all registered versions of the project model."""

    client = get_client()

    try:
        versions = client.search_model_versions(
            f"name='{MODEL_NAME}'"
        )

        for version in versions:
            print(
                f"Version: {version.version}, "
                f"Run ID: {version.run_id}, "
                f"Aliases: {version.aliases}"
            )

    except MlflowException:
        print("No registered model versions found yet.")
