import argparse
import sys
import json
import os
from pathlib import Path
from src.config.paths import paths
from kaggle_api.dataset_manager import DatasetManager
from kaggle_api.notebook_manager import NotebookManager
from kaggle_api.kernel_runner import KernelRunner

# Fix Windows console encoding
if os.name == 'nt':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def main():
    parser = argparse.ArgumentParser(description="Kaggle Training Automation")
    parser.add_argument("--push-dataset", action="store_true", help="Upload/Update dataset")
    parser.add_argument("--force-dataset", action="store_true", help="Force dataset upload even if exists")
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
        dm.push_dataset(force=args.force_dataset)

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
        output_dir = str(paths.KAGGLE_OUTPUTS)
        
        # Use subprocess to call kaggle CLI directly with live output
        import subprocess
        print(f"Downloading outputs from {notebook_slug} to {output_dir}...")
        print("(This may take a minute, please wait...)\n")
        
        try:
            # Call kaggle CLI directly without capturing output so it streams to console
            result = subprocess.run(
                ["kaggle", "kernels", "output", notebook_slug, "-p", output_dir, "--force"]
            )
            if result.returncode == 0:
                print("\n✓ Download completed successfully")
            else:
                print(f"\n✗ Download failed with exit code {result.returncode}")
                sys.exit(1)
        except Exception as e:
            print(f"\n✗ Download error: {e}")
            sys.exit(1)
        
        metrics_path = Path(output_dir) / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            print("\n=== Training Metrics ===")
            print(f"Final Loss: {metrics.get('final_loss', 'N/A')}")
            print(f"Total Epochs: {len(metrics.get('history', []))}")
            if metrics.get('history'):
                print("\nLast 5 Epochs:")
                for entry in metrics['history'][-5:]:
                    print(f"  Epoch {entry['epoch']}: train_loss={entry['train_loss']:.4f}, val_loss={entry['val_loss']:.4f}")
        else:
            print("metrics.json not found in outputs.")
            if Path(output_dir).exists():
                print(f"Files in {output_dir}:")
                for f in Path(output_dir).iterdir():
                    print(f"  - {f.name}")

if __name__ == "__main__":
    main()
