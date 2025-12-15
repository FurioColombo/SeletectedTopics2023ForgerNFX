import os
import json
import time
from pathlib import Path
from .auth import get_api_client

class NotebookManager:
    def __init__(self, config_path: str = "config/kaggle_train.yaml"):
        self.api = get_api_client()
        self.config_path = Path(config_path)
        
    def _load_config(self):
        import yaml
        if not self.config_path.exists():
            raise FileNotFoundError(f"Training config not found at {self.config_path}")
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def generate_notebook(self, dataset_slug: str, repo_url: str):
        """Generates the notebook content (kernel-metadata.json and notebook.ipynb)."""
        config = self._load_config()
        notebook_slug = config.get('notebook_slug', 'forger-nfx-training')
        title = config.get('notebook_title', 'Forger NFX Training')
        
        # Metadata
        metadata = {
            "id": f"{os.environ.get('KAGGLE_USERNAME')}/{notebook_slug}",
            "title": title,
            "code_file": "notebook.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "dataset_sources": [dataset_slug],
            "competition_sources": [],
            "kernel_sources": []
        }
        
        # Notebook Content
        # We need to install dependencies, clone repo, and run training
        # Note: We assume the repo is public or we have a way to access it.
        # If private, we might need to bundle the code in the dataset.
        # For now, assuming public repo or bundled code.
        # Actually, user said "clones the GitHub repo".
        
        repo_branch = config.get('repo_branch', 'master')
        
        # Load W&B Key from local credentials for injection (Bypasses flaky Kaggle Secrets)
        wandb_key = None
        wandb_entity = None
        creds_path = Path("credentials/wandb.json")
        if creds_path.exists():
            try:
                with open(creds_path, 'r') as f:
                    creds = json.load(f)
                    wandb_key = creds.get("api_key")
                    wandb_entity = creds.get("entity")
            except Exception as e:
                print(f"Warning: Could not read wandb.json: {e}")

        cells = []
        
        # Inject API Key cell if available
        if wandb_key:
            env_vars = [ "import os\n" ]
            env_vars.append(f"os.environ['WANDB_API_KEY'] = '{wandb_key}'\n")
            if wandb_entity:
                env_vars.append(f"os.environ['WANDB_ENTITY'] = '{wandb_entity}'\n")
            
            env_vars.append("print('✅ Injected W&B Credentials from build environment')\n")
            
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": env_vars
            })

        cells.extend([
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    f"!git clone --branch {repo_branch} --single-branch " + repo_url + " /kaggle/working/repo\n",
                    "%cd /kaggle/working/repo\n",
                    "!pip install -q -r requirements.txt\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import yaml\n",
                    "# Inject Kaggle Config\n",
                    "kaggle_config = " + json.dumps(config) + "\n",
                    "with open('config/kaggle_train.yaml', 'w') as f:\n",
                    "    yaml.dump(kaggle_config, f)\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Run Training\n",
                    f"!python train.py --dataset_root {config.get('data_root', '/kaggle/input/egfx-dataset')} "
                    f"--target_folder TubeScreamer --epochs {config.get('epochs', 50)} "
                    f"--batch_size {config.get('batch_size', 16)} "
                    f"--metrics_out {config.get('metrics_out', '/kaggle/working/metrics.json')}\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Run Evaluation\n",
                    "# Auto-discovers latest checkpoint and logs 5 samples to W&B\n",
                    f"!python evaluate.py --effect TubeScreamer --limit-samples 5 --save-preds --dataset-root {config.get('data_root', '/kaggle/input/egfx-dataset')}\n"
                ]
            }
        ])
        
        notebook = {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                    },
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.7.12"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 5
        }
        
        return metadata, notebook

    def push_notebook(self, dataset_slug: str, repo_url: str):
        """Pushes the notebook to Kaggle."""
        metadata, notebook = self.generate_notebook(dataset_slug, repo_url)
        
        # Write files temporarily
        work_dir = Path("kaggle_build")
        work_dir.mkdir(exist_ok=True)
        
        with open(work_dir / "kernel-metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        with open(work_dir / "notebook.ipynb", 'w') as f:
            json.dump(notebook, f, indent=2)
            
        print("Pushing notebook...")
        self.api.kernels_push(str(work_dir))
        print("Notebook pushed.")
        return metadata['id']
