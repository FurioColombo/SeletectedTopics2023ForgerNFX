import argparse
import sys
import json
from pathlib import Path
from kaggle_api.dataset_manager import DatasetManager
from kaggle_api.notebook_manager import NotebookManager
from kaggle_api.kernel_runner import KernelRunner

def main():
    parser = argparse.ArgumentParser(description="Kaggle Training Automation")
    parser.add_argument("--push-dataset", action="store_true", help="Upload/Update dataset")
    parser.add_argument("--run", action="store_true", help="Push and run notebook")
    parser.add_argument("--pull-metrics", action="store_true", help="Download metrics and logs")
    args = parser.parse_args()

    # Load configs
    try:
        with open("config/dataset.yaml", 'r') as f:
            import yaml
            dataset_config = yaml.safe_load(f)
        with open("config/kaggle_train.yaml", 'r') as f:
            train_config = yaml.safe_load(f)
    except FileNotFoundError as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    dataset_slug = dataset_config['slug']
    notebook_slug = f"{dataset_slug.split('/')[0]}/{train_config['notebook_slug']}"
    repo_url = train_config.get('repo_url', '')

    if args.push_dataset:
        print("=== Pushing Dataset ===")
        dm = DatasetManager()
        dm.push_dataset()

    if args.run:
        print("=== Running Notebook ===")
        nm = NotebookManager()
        nm.push_notebook(dataset_slug, repo_url)
        
        kr = KernelRunner()
        status = kr.poll_status(notebook_slug)
        print(f"Final Status: {status}")
        
        if status != 'complete':
            print("Run failed or cancelled.")
            sys.exit(1)

    if args.pull_metrics:
        print("=== Pulling Metrics ===")
        kr = KernelRunner()
        output_dir = "outputs/kaggle_runs"
        kr.download_outputs(notebook_slug, output_dir)
        
        metrics_path = Path(output_dir) / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            print("\n=== Training Metrics ===")
            print(json.dumps(metrics, indent=2))
        else:
            print("metrics.json not found in outputs.")

if __name__ == "__main__":
    main()
