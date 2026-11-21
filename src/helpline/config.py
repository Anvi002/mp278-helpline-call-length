
import yaml
from pathlib import Path


# Find the main project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Location of params.yaml
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def load_config():
    """Load project configuration from params.yaml."""

    if not PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {PARAMS_PATH}"
        )

    with open(PARAMS_PATH, "r") as file:
        config = yaml.safe_load(file)

    return config


def get_config():
    """Return the loaded project configuration."""

    return load_config()
