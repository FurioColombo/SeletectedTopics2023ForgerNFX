import os
from pathlib import Path

# Local Paths
PROJECT_ROOT = Path(__file__).parent.parent
LOCAL_DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "kaggle_runs"

# Kaggle Paths
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_DIR = Path("/kaggle/working")

def get_kaggle_dataset_path(dataset_slug: str) -> Path:
    """Returns the path where a dataset is mounted in Kaggle kernel."""
    # Dataset slug is usually "username/dataset-name"
    # Kaggle mounts it at /kaggle/input/dataset-name
    dataset_name = dataset_slug.split("/")[-1]
    return KAGGLE_INPUT_ROOT / dataset_name
