# Kaggle Training Automation

This project includes full automation for training on Kaggle GPUs.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Kaggle Credentials**:
    - Place your `kaggle.json` in `~/.kaggle/kaggle.json`.
    - Or set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.

## Configuration

-   `config/dataset.yaml`: Defines the dataset to upload.
-   `config/kaggle_train.yaml`: Defines training parameters for the remote run.

## Usage

The `kaggle_train.py` script handles the entire workflow.

### 1. Upload/Update Dataset
Uploads the local dataset defined in `config/dataset.yaml` to Kaggle.
```bash
python kaggle_train.py --push-dataset
```

### 2. Run Training
Creates a notebook on Kaggle, clones this repo, and runs `train.py`.
```bash
python kaggle_train.py --run
```

### 3. Get Results
Downloads `metrics.json` and logs from the latest run.
```bash
python kaggle_train.py --pull-metrics
```

## How it Works

1.  **Dataset**: The `kaggle_api.dataset_manager` zips your local data and pushes it to Kaggle.
2.  **Notebook**: The `kaggle_api.notebook_manager` creates a kernel that:
    -   Clones your repo.
    -   Installs requirements.
    -   Runs `train.py` with the config injected.
3.  **Training**: `train.py` runs on Kaggle GPU, saves checkpoints to `/kaggle/working`, and writes `metrics.json`.
4.  **Retrieval**: `kaggle_api.kernel_runner` downloads the output files.
