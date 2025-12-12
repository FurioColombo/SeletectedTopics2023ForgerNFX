"""
Run multiple Kaggle training experiments with different hyperparameters.

This script helps you run a series of training experiments on Kaggle,
each with different configurations, then downloads and evaluates the results.
"""
import yaml
import shutil
from pathlib import Path
import time


def create_experiment_config(
    experiment_name: str,
    learning_rate: float,
    epochs: int = 50,
    batch_size: int = 16
):
    """
    Create a modified kaggle_train.yaml for an experiment.
    
    Args:
        experiment_name: Name for this experiment
        learning_rate: Learning rate to use
        epochs: Number of epochs
        batch_size: Batch size
    """
    # Load base config
    with open("config/kaggle_train.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify for this experiment
    config['epochs'] = epochs
    config['batch_size'] = batch_size
    # Note: Learning rate needs to be passed to train.py via command args
    
    # Save as experiment config
    experiment_config_path = Path("config") / f"experiment_{experiment_name}.yaml"
    with open(experiment_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return experiment_config_path


def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def main():
    print_header("🧪 KAGGLE EXPERIMENT RUNNER")
    
    print("""
This script helps you run multiple training experiments on Kaggle with
different hyperparameters (learning rates, batch sizes, etc.)

WORKFLOW:
1. Define your experiments below
2. For each experiment:
   - Push code & dataset to Kaggle
   - Run training
   - Download trained model
3. Evaluate all models locally with evaluate.py

Let's define your experiments!
""")
    
    # Define experiments
    experiments = []
    
    print("\n" + "-" * 70)
    print("EXPERIMENT CONFIGURATION")
    print("-" * 70)
    
    # Experiment 1
    print("\n📝 Experiment 1: Baseline")
    exp1_lr = float(input("  Learning rate (default 0.01): ").strip() or "0.01")
    exp1_epochs = int(input("  Epochs (default 50): ").strip() or "50")
    experiments.append({
        'name': 'baseline',
        'lr': exp1_lr,
        'epochs': exp1_epochs,
        'batch_size': 16
    })
    
    # Experiment 2
    print("\n📝 Experiment 2: Different learning rate")
    exp2_lr = float(input("  Learning rate (try 0.001 or 0.1): ").strip() or "0.001")
    exp2_epochs = int(input("  Epochs (default 50): ").strip() or "50")
    experiments.append({
        'name': 'lr_experiment',
        'lr': exp2_lr,
        'epochs': exp2_epochs,
        'batch_size': 16
    })
    
    # Add more?
    while True:
        add_more = input("\n➕ Add another experiment? (y/N): ").strip().lower()
        if add_more != 'y':
            break
        
        exp_name = input("  Experiment name: ").strip().replace(' ', '_')
        exp_lr = float(input("  Learning rate: ").strip())
        exp_epochs = int(input("  Epochs: ").strip())
        exp_batch = int(input("  Batch size (default 16): ").strip() or "16")
        
        experiments.append({
            'name': exp_name,
            'lr': exp_lr,
            'epochs': exp_epochs,
            'batch_size': exp_batch
        })
    
    # Review experiments
    print_header("📋 EXPERIMENT SUMMARY")
    for i, exp in enumerate(experiments, 1):
        print(f"\n  {i}. {exp['name']}")
        print(f"     Learning Rate: {exp['lr']}")
        print(f"     Epochs:        {exp['epochs']}")
        print(f"     Batch Size:    {exp['batch_size']}")
    
    proceed = input("\n\nProceed with these experiments? (Y/n): ").strip().lower()
    if proceed == 'n':
        print("Aborted.")
        return
    
    # Instructions for running
    print_header("🚀 RUNNING EXPERIMENTS")
    
    print("""
TO RUN EACH EXPERIMENT ON KAGGLE:

For each experiment, you need to:
1. Modify train.py call in your Kaggle notebook to use the learning rate
2. Push and run the notebook
3. Download results

Here's what to do:
""")
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n" + "─" * 70)
        print(f"EXPERIMENT {i}: {exp['name']}")
        print("─" * 70)
        
        print(f"\n1️⃣  In your Kaggle notebook, modify the training command:")
        print(f"   ```python")
        print(f"   !python train.py \\")
        print(f"       --dataset_root /kaggle/input/egfx-dataset/EGFxDataset \\")
        print(f"       --target_folder TubeScreamer \\")
        print(f"       --lr {exp['lr']} \\")
        print(f"       --epochs {exp['epochs']} \\")
        print(f"       --batch_size {exp['batch_size']} \\")
        print(f"       --metrics_out /kaggle/working/metrics.json")
        print(f"   ```")
        
        print(f"\n2️⃣  Run on Kaggle:")
        print(f"   python kaggle_train.py --push-dataset --run")
        
        print(f"\n3️⃣  Download results:")
        print(f"   python kaggle_train.py --pull-metrics")
        
        print(f"\n4️⃣  Rename downloaded model (to avoid overwriting):")
        print(f"   Move outputs/kaggle_runs/*.pt to runs/{exp['name']}/")
        
        input(f"\n   Press Enter when Experiment {i} is complete...")
    
    print_header("✅ ALL EXPERIMENTS COMPLETE")
    
    print("""
Now evaluate all your models!

Run the evaluation script for each trained model:
    python evaluate.py

Select each checkpoint and compare results in the evaluation reports.

TIP: Look at these metrics to compare models:
  - ESR (Error Signal Ratio) - lower is better
  - Spectral Convergence - lower is better  
  - Phase Response Error - lower is better

Open the HTML reports side-by-side in your browser to compare!
""")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAborted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
