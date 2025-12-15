"""
Professional evaluation script for guitar effect models.
Supports both interactive (wizard) mode and non-interactive CLI mode for automation.
"""
import argparse
import sys
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

# Local imports
from src.config.paths import paths
from src.config.config import ProjectConfig, ModelConfig
from src.data.egfx import EGFxDataset
from src.data.loader import create_dataloaders
from src.models.lstm import LSTMModel
from src.models.conv import ConvModel
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.visualizer import MetricsVisualizer

# --- UTILS ---

def print_header(text: str):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def find_checkpoints() -> list[Path]:
    """Find all available model checkpoints in runs/ and checkpoints/."""
    checkpoints = []
    for directory in [paths.RUNS, paths.CHECKPOINTS]:
        if directory.exists():
            for model_dir in directory.iterdir():
                if model_dir.is_dir():
                    checkpoints.extend(model_dir.glob("*.pt"))
    return sorted(checkpoints, key=lambda x: x.stat().st_mtime, reverse=True)

def find_effect_folders() -> list[str]:
    """Find available effect folders in the dataset."""
    dataset_root = paths.DATASETS / "EGFxDataset" # Default location
    effects = []
    if dataset_root.exists():
        for folder in dataset_root.iterdir():
            if folder.is_dir() and folder.name != "Clean":
                effects.append(folder.name)
    return sorted(effects)

def load_model_from_checkpoint(checkpoint_path: Path, device: str):
    """Load model and config from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Infer model type
    filename = checkpoint_path.name.lower()
    if 'lstm' in filename:
        model_type = 'lstm'
    elif 'conv' in filename:
        model_type = 'conv'
    else:
        # Fallback to checking config if available, else default to lstm
        model_type = 'lstm'

    # Load Config
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
    elif 'config' in checkpoint:
         # Handle legacy where config might be stored directly or as dict
         # But usually we stored it as model_config.
         # Let's try to reconstruct if missing.
         model_config = ModelConfig(name=model_type, hidden_size=64) # Dangerous assumption but necessary for very old ckpts
    else:
        model_config = ModelConfig(name=model_type, hidden_size=64)

    # Initialize Model
    if model_type == 'lstm':
        model = LSTMModel(model_config)
    else:
        model = ConvModel(model_config)
    
    # Load State
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', 'unknown')
    loss = checkpoint.get('loss', 'unknown')
    
    return model, model_config, {'epoch': epoch, 'loss': loss, 'type': model_type}

# --- INTERACTIVE MODES ---

def interactive_select_checkpoint() -> Path:
    print_header("📦 SELECT MODEL CHECKPOINT")
    checkpoints = find_checkpoints()
    if not checkpoints:
        print("❌ No checkpoints found.")
        sys.exit(1)

    print("\nAvailable checkpoints (most recent first):\n")
    for i, ckpt in enumerate(checkpoints, 1):
        size_mb = ckpt.stat().st_size / (1024 * 1024)
        modified = datetime.fromtimestamp(ckpt.stat().st_mtime)
        print(f"  [{i}] {ckpt.parent.name}/{ckpt.name}  ({size_mb:.1f} MB, {modified})")

    while True:
        try:
            choice = input("\nSelect checkpoint number (or 'q'): ").strip()
            if choice.lower() == 'q': sys.exit(0)
            idx = int(choice) - 1
            if 0 <= idx < len(checkpoints):
                return checkpoints[idx]
        except ValueError:
            pass

def interactive_select_effect() -> str:
    print_header("🎸 SELECT GUITAR EFFECT")
    effects = find_effect_folders()
    
    if effects:
        for i, effect in enumerate(effects, 1):
            print(f"  [{i}] {effect}")
    print(f"  [0] Custom Name")

    while True:
        try:
            choice = input("\nSelect number: ").strip()
            if choice == '0':
                return input("Enter effect name: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(effects):
                return effects[idx]
        except ValueError:
            pass

# --- MAIN EVALUATION LOGIC ---

def run_evaluation(
    checkpoint_path: Path,
    effect_name: str,
    device: str,
    save_predictions: bool,
    limit_samples: Optional[int] = None,
    dataset_root: Optional[Path] = None,
    notes: str = ""
):
    """
    Main execution pipeline for evaluation.
    """
    # 1. Setup Session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = checkpoint_path.parent.name # Use run name as model identifier
    session_id = f"{timestamp}_{model_name}_eval"
    session_dir = paths.OUTPUTS / "evaluations" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    print_header(f"🚀 STARTING EVALUATION: {session_id}")
    print(f"Model: {checkpoint_path}")
    print(f"Effect: {effect_name}")
    print(f"Device: {device}")
    
    # 2. Load Model
    try:
        model, _, info = load_model_from_checkpoint(checkpoint_path, device)
        print(f"✅ Model loaded (Epoch {info['epoch']}, Loss {info['loss']})")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        # Traceback for debugging
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 3. Load Data
    if dataset_root is None:
        dataset_root = paths.DATASETS / "EGFxDataset"
    
    input_root = dataset_root / "Clean"
    target_root = dataset_root / effect_name
    
    if not target_root.exists():
        print(f"❌ Target folder not found: {target_root}")
        sys.exit(1)

    print(f"Loading data from {dataset_root}...")
    dataset = EGFxDataset(
        input_root=str(input_root),
        output_root=str(target_root),
        block_size=2048, # Standard evaluation block size
        sample_rate=44100
    )
    
    # Create Test Loader (reuse existing project config mostly for batch size/workers if needed, or defaults)
    # Using simple defaults for evaluation to ensure robustness
    test_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=16, 
        shuffle=False, 
        num_workers=0 # Safer for some environments
    )
    print(f"✅ Dataset loaded: {len(dataset)} samples")

    # 4. Evaluate
    evaluator = ModelEvaluator(model, test_loader, device=device, sample_rate=44100)
    
    # Determine output directory for wavs
    wav_dir = session_dir / "predictions" if save_predictions else None
    
    # Note: evaluator.evaluate might need updates to accept limit_samples if not already there, 
    # but we can implement it in the next step or assume it's there. 
    # Checking previous file view, 'evaluate' didn't have 'limit_samples'. We need to add it!
    # For now, we pass it, assuming I will update evaluator.py next.
    
    # To avoid runtime error before update, let's check signatures or just rely on the plan order.
    # Plan says: "Update src/evaluation/evaluator.py" is step 3. 
    # So I should update evaluator.py quickly after this.
    try:
        results = evaluator.evaluate(
            save_predictions=save_predictions,
            output_dir=wav_dir,
            limit_samples=limit_samples 
        )
    except TypeError:
        # Fallback if I haven't updated evaluator.py yet
        print("⚠️ 'limit_samples' not supported in current evaluator.py, ignoring.")
        results = evaluator.evaluate(
            save_predictions=save_predictions,
            output_dir=wav_dir
        )

    # 5. Save Results
    evaluator.save_results(results, session_dir / "results.json", format='json')
    evaluator.save_results(results, session_dir / "results.csv", format='csv')

    # 6. Generate Report
    print("Generating HTML report...")
    visualizer = MetricsVisualizer(results, sample_rate=44100)
    
    # Get a sample for visualization (first batch)
    inputs, targets = next(iter(test_loader))
    inputs, targets = inputs.to(device), targets.to(device)
    with torch.no_grad():
        preds = model(inputs)
    
    visualizer.create_full_report(
        output_path=session_dir / "report.html",
        sample_audio=(inputs[0].cpu(), preds[0].cpu(), targets[0].cpu())
    )

    # 7. Metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(checkpoint_path),
        "effect": effect_name,
        "notes": notes,
        "device": device,
        "metrics_summary": {k: v['mean'] for k, v in results['metrics'].items() if isinstance(v, dict)}
    }
    with open(session_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print_header("✅ EVALUATION COMPLETED")
    print(f"Results saved to: {session_dir}")
    if 'summary' in results:
        print(results['summary'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Guitar Effect Models")
    
    # Modes
    parser.add_argument("--interactive", action="store_true", help="Run in interactive wizard mode")
    
    # Automation Arguments
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint (.pt)")
    parser.add_argument("--effect", type=str, help="Name of the target effect folder (e.g. TubeScreamer)")
    parser.add_argument("--dataset-root", type=str, default=None, help="Root path of dataset")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    
    # Flags
    parser.add_argument("--save-preds", action="store_true", help="Save all predicted audio files to disk (Warning: Large space usage)")
    parser.add_argument("--limit-samples", type=int, default=None, help="Limit number of samples to process (for debugging)")
    parser.add_argument("--notes", type=str, default="", help="Optional notes for this run")

    args = parser.parse_args()

    # LOGIC:
    # If interactive flag IS set, or NO args provided -> Interactive
    # If checkpoint/effect provided -> Automation
    
    is_interactive = args.interactive or (not args.checkpoint and not args.effect)

    if is_interactive:
        ckpt = interactive_select_checkpoint()
        eff = interactive_select_effect()
        
        # Simple interactive config
        dev = "cuda" if torch.cuda.is_available() and input("Use GPU? (Y/n): ").lower() != 'n' else "cpu"
        save = input("Save predictions? (y/N): ").lower() == 'y'
        notes = input("Notes: ").strip()
        
        run_evaluation(ckpt, eff, dev, save, notes=notes)
    else:
        # Automation Mode checks
        if not args.checkpoint:
            # Auto-find latest if not specified? Or error? 
            # Let's try to auto-find latest best_model if in a run context, otherwise error.
            print("No checkpoint specified, looking for latest 'best_model.pt'...")
            checkpoints = find_checkpoints()
            if checkpoints:
                args.checkpoint = str(checkpoints[0])
                print(f"Auto-selected: {args.checkpoint}")
            else:
                print("❌ No checkpoint found provided and none found automatically.")
                sys.exit(1)
        
        if not args.effect:
            # Error out, we need to know what to evaluate against
            print("❌ --effect argument is required in non-interactive mode.")
            sys.exit(1)
            
        run_evaluation(
            checkpoint_path=Path(args.checkpoint),
            effect_name=args.effect,
            device=args.device,
            save_predictions=args.save_preds,
            limit_samples=args.limit_samples,
            dataset_root=Path(args.dataset_root) if args.dataset_root else None,
            notes=args.notes
        )
