import os
import json
from pathlib import Path
from typing import Optional

def authenticate():
    """
    Authenticates with Kaggle API.
    Checks for KAGGLE_USERNAME and KAGGLE_KEY env vars,
    or ~/.kaggle/kaggle.json.
    """
    # Check environment variables
    if "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ:
        print("Authenticated via environment variables.")
        return

    # Check local config in project root
    local_config = Path("kaggle.json")
    if local_config.exists():
        print(f"Found local kaggle.json at {local_config}")
        with open(local_config, 'r') as f:
            config = json.load(f)
            os.environ["KAGGLE_USERNAME"] = config["username"]
            os.environ["KAGGLE_KEY"] = config["key"]
        return

    # Check local config in home dir
    kaggle_config_path = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_config_path.exists():
        with open(kaggle_config_path, 'r') as f:
            config = json.load(f)
            os.environ["KAGGLE_USERNAME"] = config["username"]
            os.environ["KAGGLE_KEY"] = config["key"]
        print(f"Authenticated via {kaggle_config_path}")
        return

    raise RuntimeError(
        "Kaggle credentials not found. "
        "Please place 'kaggle.json' in this directory, or in ~/.kaggle/, "
        "or set KAGGLE_USERNAME and KAGGLE_KEY environment variables."
    )

def get_api_client():
    """Returns an authenticated Kaggle API client."""
    authenticate()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api
