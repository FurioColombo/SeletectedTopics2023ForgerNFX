"""
Interactive evaluation script with user-friendly interface.

Provides a guided interface to evaluate guitar effect models without
memorizing command-line arguments.
"""
import torch
from pathlib import Path
from datetime import datetime
import json
import sys

from src.config.paths import paths
from src.models.lstm import LSTMModel
from src.models.conv import ConvModel
from src.config.config import ModelConfig, ProjectConfig
from src.data.egfx import EGFxDataset
from src.data.loader import create_dataloaders
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.visualizer import MetricsVisualizer


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def find_checkpoints():
    """Find all available model checkpoints."""
    checkpoints = []
    
    # Search in runs directory
    if paths.RUNS.exists():
        for model_dir in paths.RUNS.iterdir():
            if model_dir.is_dir():
                # Look for .pt files
                for ckpt in model_dir.glob("*.pt"):
                    checkpoints.append(ckpt)
    
    # Search in checkpoints directory
    if paths.CHECKPOINTS.exists():
        for model_dir in paths.CHECKPOINTS.iterdir():
            if model_dir.is_dir():
                for ckpt in model_dir.glob("*.pt"):
                    checkpoints.append(ckpt)
    
    return sorted(checkpoints, key=lambda x: x.stat().st_mtime, reverse=True)


def select_checkpoint():
    """Interactive checkpoint selection."""
    print_header("📦 SELECT MODEL CHECKPOINT")
    
    checkpoints = find_checkpoints()
    
    if not checkpoints:
        print("\n❌ No checkpoints found in runs/ or checkpoints/")
        print("   Train a model first using train.py")
        sys.exit(1)
    
    print("\nAvailable checkpoints (most recent first):\n")
    for i, ckpt in enumerate(checkpoints, 1):
        # Get file info
        size_mb = ckpt.stat().st_size / (1024 * 1024)
        modified = datetime.fromtimestamp(ckpt.stat().st_mtime)
        
        print(f"  [{i}] {ckpt.parent.name}/{ckpt.name}")
        print(f"      Size: {size_mb:.1f} MB  |  Modified: {modified.strftime('%Y-%m-%d %H:%M')}")
        print()
    
    while True:
        try:
            choice = input("\nSelect checkpoint number (or 'q' to quit): ").strip()
            if choice.lower() == 'q':
                print("Aborted.")
                sys.exit(0)
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(checkpoints):
                selected = checkpoints[choice_num - 1]
                print(f"\n✅ Selected: {selected.parent.name}/{selected.name}")
                return selected
            else:
                print(f"Please enter a number between 1 and {len(checkpoints)}")
        except ValueError:
            print("Invalid input. Please enter a number.")


def find_effect_folders():
    """Find available effect folders in dataset."""
    effects = []
    dataset_root = paths.DATASETS / "EGFxDataset"
    
    if dataset_root.exists():
        for folder in dataset_root.iterdir():
            if folder.is_dir() and folder.name != "Clean":
                effects.append(folder.name)
    
    return sorted(effects)


def select_effect():
    """Interactive effect selection."""
    print_header("🎸 SELECT GUITAR EFFECT TO EVALUATE")
    
    print("\nWhat effect was this model trained to emulate?")
    print("(e.g., TubeScreamer, BigMuff, RAT, etc.)\n")
    
    effects = find_effect_folders()
    
    if effects:
        print("Available effects in your dataset:")
        for i, effect in enumerate(effects, 1):
            print(f"  [{i}] {effect}")
        print(f"  [0] Enter custom name")
        
        while True:
            try:
                choice = input("\nSelect effect number: ").strip()
                choice_num = int(choice)
                
                if choice_num == 0:
                    effect = input("Enter effect name: ").strip()
                    if effect:
                        return effect
                elif 1 <= choice_num <= len(effects):
                    return effects[choice_num - 1]
                else:
                    print(f"Please enter 0-{len(effects)}")
            except ValueError:
                print("Invalid input. Please enter a number.")
    else:
        effect = input("\nEnter effect name (e.g., TubeScreamer): ").strip()
        return effect


def configure_evaluation():
    """Interactive configuration."""
    print_header("⚙️  CONFIGURE EVALUATION")
    
    config = {}
    
    # Device selection
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        print("\n🖥️  GPU detected!")
        use_gpu = input("Use GPU for evaluation? (Y/n): ").strip().lower()
        config['device'] = 'cuda' if use_gpu != 'n' else 'cpu'
    else:
        print("\n🖥️  Using CPU (no GPU detected)")
        config['device'] = 'cpu'
    
    # Notes
    print("\n📝 Optional: Add notes about this evaluation")
    print("   (e.g., 'After 100 epochs', 'Testing hyperparameter changes')")
    notes = input("Notes (or press Enter to skip): ").strip()
    config['notes'] = notes
    
    # Save predictions (DISABLED by default to save memory)
    print("\n💾 Save predicted audio samples?")
    print("   ⚠️  WARNING: This creates many .wav files and uses significant disk space")
    save_pred = input("Save predictions? (y/N): ").strip().lower()
    config['save_predictions'] = (save_pred == 'y')
    
    return config


def create_evaluation_session(model_name, effect_name, notes=""):
    """Create timestamped evaluation session."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"{timestamp}_{model_name}_{effect_name}"
    
    session_dir = paths.OUTPUTS / "evaluations" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "effect_name": effect_name,
        "notes": notes,
    }
    
    with open(session_dir / "session_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return session_dir


def load_model_from_checkpoint(checkpoint_path):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Infer model type from checkpoint or filename
    if 'lstm' in checkpoint_path.name.lower():
        model_type = 'lstm'
    elif 'conv' in checkpoint_path.name.lower():
        model_type = 'conv'
    else:
        model_type = 'lstm'  # Default
    
    # Create model config
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
    else:
        model_config = ModelConfig(name=model_type, hidden_size=64)
    
    # Create and load model
    if model_type == 'lstm':
        model = LSTMModel(model_config)
    else:
        model = ConvModel(model_config)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', 'unknown'),
        'loss': checkpoint.get('loss', 'unknown'),
        'type': model_type
    }
    
    return model, model_config, checkpoint_info


def main():
    print_header("🎸 GUITAR EFFECT MODEL EVALUATION")
    
    # 1. Select checkpoint
    checkpoint_path = select_checkpoint()
    
    # 2. Select effect
    effect_name = select_effect()
    
    # 3. Configure evaluation
    config = configure_evaluation()
    
    # 4. Confirm
    print_header("📋 EVALUATION SUMMARY")
    print(f"\n  Checkpoint: {checkpoint_path.parent.name}/{checkpoint_path.name}")
    print(f"  Effect:     {effect_name}")
    print(f"  Device:     {config['device'].upper()}")
    print(f"  Save WAVs:  {'Yes' if config['save_predictions'] else 'No (saves memory)'}")
    if config['notes']:
        print(f"  Notes:      {config['notes']}")
    
    proceed = input("\n\nProceed with evaluation? (Y/n): ").strip().lower()
    if proceed == 'n':
        print("Aborted.")
        return
    
    # 5. Create session
    print_header("🚀 RUNNING EVALUATION")
    model_name = checkpoint_path.stem
    session_dir = create_evaluation_session(model_name, effect_name, config['notes'])
    print(f"\n📁 Session: {session_dir.name}")
    
    # 6. Load model
    print("\n⏳ Loading model...")
    try:
        model, model_config, ckpt_info = load_model_from_checkpoint(checkpoint_path)
        print(f"✅ Loaded {ckpt_info['type'].upper()} model (epoch {ckpt_info['epoch']})")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # 7. Load dataset
    print("\n⏳ Loading dataset...")
    dataset_root = paths.DATASETS / "EGFxDataset"
    input_root = dataset_root / "Clean"
    target_root = dataset_root / effect_name
    
    if not target_root.exists():
        print(f"❌ Effect folder not found: {target_root}")
        print(f"   Available folders: {', '.join(find_effect_folders())}")
        return
    
    try:
        dataset = EGFxDataset(
            input_root=str(input_root),
            output_root=str(target_root),
            block_size=2048,
            sample_rate=44100
        )
        print(f"✅ Loaded {len(dataset)} audio pairs")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    project_config = ProjectConfig()
    train_loader, val_loader, test_loader = create_dataloaders(dataset, project_config)
    
    # 8. Run evaluation
    print(f"\n⏳ Evaluating on {len(test_loader.dataset)} test samples...")
    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        device=config['device'],
        sample_rate=44100
    )
    
    predictions_dir = session_dir / "predictions" if config['save_predictions'] else None
    results = evaluator.evaluate(
        save_predictions=config['save_predictions'],
        output_dir=predictions_dir
    )
    
    # 9. Save results
    print("\n⏳ Saving results...")
    evaluator.save_results(results, session_dir / "results.json", format='json')
    evaluator.save_results(results, session_dir / "results.csv", format='csv')
    
    # 10. Create visualizations
    print("⏳ Creating interactive report...")
    visualizer = MetricsVisualizer(results, sample_rate=44100)
    
    # Get sample for visualization
    sample_batch = next(iter(test_loader))
    inputs, targets = sample_batch
    inputs = inputs.to(config['device'])
    targets = targets.to(config['device'])
    
    with torch.no_grad():
        predictions = model(inputs)
    
    report_path = session_dir / "evaluation_report.html"
    visualizer.create_full_report(
        output_path=report_path,
        sample_audio=(inputs[0].cpu(), predictions[0].cpu(), targets[0].cpu())
    )
    
    # 11. Save metadata
    eval_metadata = {
        "checkpoint": str(checkpoint_path),
        "effect": effect_name,
        "device": config['device'],
        "model_type": ckpt_info['type'],
        "test_samples": len(test_loader.dataset),
        "metrics_summary": {k: v['mean'] for k, v in results['metrics'].items()}
    }
    
    with open(session_dir / "evaluation_metadata.json", 'w') as f:
        json.dump(eval_metadata, f, indent=2)
    
    # 12. Print results summary
    print(results['summary'])
    
    print_header("✅ EVALUATION COMPLETE")
    print(f"\n📁 Results saved to: {session_dir.name}")
    print(f"\n📄 Generated files:")
    print(f"   • evaluation_report.html  - 🌐 Interactive dashboard")
    print(f"   • results.json           - Detailed metrics")
    print(f"   • results.csv            - Metrics spreadsheet")
    print(f"   • session_metadata.json  - Session info")
    if config['save_predictions']:
        print(f"   • predictions/           - Audio samples")
    
    print(f"\n🌐 Open the interactive report:")
    print(f"   {report_path.absolute()}")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAborted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
