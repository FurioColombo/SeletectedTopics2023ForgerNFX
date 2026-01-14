import warnings
import os

# Suppress pydantic warnings BEFORE any imports that might trigger them
warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Suppress W&B verbose output
os.environ['WANDB_SILENT'] = 'true'

import argparse
import torch
from pathlib import Path
import yaml
import wandb
import sys
from datetime import datetime

from src.logging.wandb_logger import WandbLogger
from src.models.lstm import LSTMModel
from src.models.conv import ConvModel
from src.data.egfx import EGFxDataset
from src.inference.runners import SegmentRunner, SequenceRunner
from src.evaluation.analyzer import MetricAnalyzer
from src.evaluation.metrics import calculate_esr, calculate_mse, phase_response_error

from src.evaluation.visualizer import MetricsVisualizer
from src.utils.gpu import find_optimal_batch_size, print_gpu_info

def load_config(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_evaluation(
    checkpoint_path: Path,
    dataset_root: Path,
    effect_name: str,
    output_dir: Path,
    device: str = "cpu",
    limit_samples: int = None,  # Can be int (sample count) or float (fraction)
    save_preds: bool = False
):
    print(f"\n🚀 STARTING MODULAR EVALUATION")
    print(f"Model: {checkpoint_path}")
    print(f"Dataset: {dataset_root}")
    
    # Print GPU info
    print_gpu_info()
    
    # Initialize logger first
    logger = WandbLogger(
        project="forger-nfx",
        config={"checkpoint": str(checkpoint_path), "effect": effect_name},
        name=f"eval_{effect_name}_{datetime.now().strftime('%H%M')}",
        tags=["modular-eval"]
    )
    
    # 1. Load Model
    print("Loading model...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Infer architecture
    from src.config.config import ModelConfig
    
    # Extract state_dict and config
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        saved_config = checkpoint.get('config', {})
    else:
        state_dict = checkpoint
        saved_config = {}

    if 'conv' in str(checkpoint_path).lower():
         # Placeholder for ConvModel config if needed
         model = ConvModel(ModelConfig(name="conv")) 
    else:
        # LSTM Initialization
        # Try to get hidden_size from saved config
        if saved_config and 'hidden_size' in saved_config:
            hidden_size = saved_config['hidden_size']
        else:
            # Fallback: Infer from state_dict shape
            hidden_size = 16 # fallback
            if 'lstm.weight_hh_l0' in state_dict:
                hidden_size = state_dict['lstm.weight_hh_l0'].shape[1]
        
        print(f"Initializing LSTM with hidden_size={hidden_size}")
        config = ModelConfig(name="lstm", hidden_size=hidden_size)
        model = LSTMModel(config)
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # 2. Setup Data
    input_folder = "Clean" # Default, could be argument
    input_path = dataset_root / input_folder
    target_path = dataset_root / effect_name
    
    print(f"Data Input: {input_path}")
    print(f"Data Target: {target_path}")

    # 3. PHASE 1: Quantitative (Segments)
    print("\n📊 Phase 1: Quantitative Evaluation (Segments)")
    try:
        dataset = EGFxDataset(
            input_root=str(input_path),
            output_root=str(target_path),
            block_size=2048
        )
    except ValueError as e:
        print(f"❌ Dataset Init Failed: {e}")
        print(f"Please check if '{input_folder}' and '{effect_name}' folders exist in {dataset_root}")
        sys.exit(1)

    # Find optimal batch size dynamically
    def test_batch(batch_size: int):
        """Test function for batch size selection."""
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0)
        batch = next(iter(test_loader))
        inputs = batch[0].to(device)
        with torch.no_grad():
            _ = model(inputs)
    
    if device == "cuda":
        optimal_batch_size = find_optimal_batch_size(
            test_fn=test_batch,
            min_batch_size=8,
            max_batch_size=256,
            device=device
        )
    else:
        optimal_batch_size = 32  # Default for CPU
        print(f"ℹ️  Using default CPU batch size: {optimal_batch_size}")

    # Optimize num_workers based on CPU cores
    import os
    cpu_count = os.cpu_count() or 2
    # For I/O-bound tasks (loading audio files), use more workers
    # Kaggle has ~2 cores, local machines might have 8+
    optimal_workers = min(cpu_count * 2, 4) if device == "cuda" else 2
    
    print(f"📦 DataLoader config: batch_size={optimal_batch_size}, num_workers={optimal_workers}")

    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=optimal_batch_size, 
        num_workers=optimal_workers,
        pin_memory=(device == "cuda"),
        prefetch_factor=4 if optimal_workers > 0 else None,  # Prefetch 4 batches per worker
        persistent_workers=(optimal_workers > 0)  # Keep workers alive between epochs
    )
    
    # Calculate how many batches to process
    total_samples = len(dataset)
    print(f"📊 Total dataset samples: {total_samples:,}")
    
    # Handle fraction (passed as float between 0 and 1)
    if limit_samples is not None and isinstance(limit_samples, float) and 0 < limit_samples < 1:
        actual_limit = int(total_samples * limit_samples)
        print(f"🎯 Using {limit_samples*100:.1f}% of dataset = {actual_limit:,} samples")
        limit_samples = actual_limit
    
    # Determine limit_batches based on limit_samples
    limit_batches = None
    if limit_samples is not None:
        limit_batches = (limit_samples // optimal_batch_size) + (1 if limit_samples % optimal_batch_size else 0)
        actual_samples = min(limit_samples, total_samples)
        print(f"🎯 Evaluating on {actual_samples:,} samples ({limit_batches} batches)")
    else:
        # Full dataset
        print(f"🎯 Evaluating on full dataset ({total_samples:,} samples)")
    
    analyzer = MetricAnalyzer()
    seg_runner = SegmentRunner(
        model, 
        dataloader, 
        device, 
        limit_batches=limit_batches
    )
    
    for batch_res in seg_runner.run():
        analyzer.process_batch(batch_res)
        
    quantitative_results = analyzer.get_aggregated_results()
    print(analyzer.generate_summary())
    
    # Log Quantitative
    logger.log_test_quantitative(quantitative_results)
    
    # 4. PHASE 2: Qualitative (Full Sequences)
    print("\n👂 Phase 2: Qualitative Evaluation (Full Sequences)")
    
    # Use the actual files from the dataset (which we know exist)
    if hasattr(dataset, 'input_files') and hasattr(dataset, 'output_files') and len(dataset.input_files) > 0:
        # Select first 3 file pairs from the dataset
        num_files = min(3, len(dataset.input_files))
        selected_input = dataset.input_files[:num_files]
        selected_output = dataset.output_files[:num_files]
        
        file_pairs = [(Path(inp), Path(out)) for inp, out in zip(selected_input, selected_output)]
        print(f"✅ Selected {len(file_pairs)} file pairs from dataset for qualitative analysis")
    else:
        print("⚠️  Dataset doesn't have file lists, trying fallback...")
        # Fallback: try to find files directly
        all_files = list(Path(input_path).glob("*.wav"))
        print(f"Found {len(all_files)} audio files in {input_path}")
        
        test_files = all_files[:min(3, len(all_files))]
        print(f"Selected {len(test_files)} files for qualitative analysis")
        
        file_pairs = [(f, Path(target_path) / f.name) for f in test_files if (Path(target_path) / f.name).exists()]
        print(f"Found {len(file_pairs)} valid file pairs (with matching targets)")
    
    if not file_pairs:
        print("⚠️  No file pairs found for qualitative evaluation, skipping...")
    else:
        print(f"✅ Processing {len(file_pairs)} file pairs for qualitative evaluation")
        
        # Switch to CPU for full sequence inference to avoid cuDNN limits on long sequences
        # (CUDNN_STATUS_NOT_SUPPORTED errors on very long LSTMs)
        print("ℹ️  Switching to CPU for qualitative inference (safer for long sequences)...")
        model.cpu()
        
        # clear GPU cache
        if device == "cuda":
            torch.cuda.empty_cache()
            
        seq_runner = SequenceRunner(model, file_pairs, device="cpu")
        visualizer = MetricsVisualizer(quantitative_results) 
        
        # Collect all qualitative results first
        qualitative_sequences = []
        
        for seq_res in seq_runner.run():
            name = seq_res['name']
            print(f"  Processing {name}...")
            
            # Generate and display spectral overlap plot INLINE
            inp_t = seq_res['input']
            tgt_t = seq_res['target']
            pred_t = seq_res['prediction']
            
            # Calculate metrics for this specific sample
            # (ensure tensors are on same device/shape for metric calc)
            # Pred/Target likely on CPU from runner
            # FAST Metric Calculation (avoid expensive cross-correlation on CPU)
            print(f"    Calculating fast metrics (ESR, MSE, Phase) for {name}...")
            sample_metrics = {
                'esr': calculate_esr(pred_t, tgt_t),
                'mse': calculate_mse(pred_t, tgt_t),
                'phase_response_error_rad': phase_response_error(pred_t, tgt_t)
            }
            
            # Store metrics in result for Logger
            seq_res['metrics'] = sample_metrics
            
            # Create the plot (now with metrics in title)
            fig = visualizer.plot_spectral_overlap(inp_t, pred_t, tgt_t, metrics=sample_metrics)
            
            # Display inline in notebook
            print(f"\n📊 Spectral Overlap for {name}:")
            try:
                fig.show()  # This will render in Kaggle notebook
            except Exception as e:
                print(f"⚠️  Could not display plot inline: {e}")
            
            # Save WAVs locally
            if save_preds:
                (output_dir / "predictions").mkdir(exist_ok=True, parents=True)
                import torchaudio
                # Ensure 2D (channels, time)
                pred_wav = seq_res['prediction']
                if pred_wav.dim() == 1:
                    pred_wav = pred_wav.unsqueeze(0)
                elif pred_wav.dim() == 3:
                     pred_wav = pred_wav.squeeze(0)
                
                torchaudio.save(output_dir / "predictions" / f"{name}_pred.wav", pred_wav, seq_res['sample_rate'])
            
            qualitative_sequences.append(seq_res)
        
        print(f"\n✅ Collected {len(qualitative_sequences)} qualitative sequences")
        
        # Log Qualitative (Batch upload to Table)
        if qualitative_sequences:
            print("📤 Logging qualitative results to W&B...")
            logger.log_test_qualitative(qualitative_sequences, visualizer=visualizer)
            print("✅ Qualitative logging complete")
        else:
            print("⚠️  No qualitative sequences to log")
    
    logger.finish()
    print("\n✅ Evaluation Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on audio effect dataset")
    parser.add_argument("--checkpoint", type=str, required=False, help="Path to model checkpoint. Auto-discovered if not provided.")
    parser.add_argument("--effect", type=str, required=True, help="Effect name (e.g., TubeScreamer)")
    parser.add_argument("--dataset-root", type=str, required=True, help="Root directory of dataset")
    parser.add_argument("--save-preds", action="store_true", help="Save predicted audio files")
    parser.add_argument("--limit-samples", type=int, default=None, help="Limit to N samples (overrides --fraction)")
    parser.add_argument("--fraction", type=float, default=None, help="Evaluate on fraction of dataset (e.g., 0.05 for 5%%)")
    
    args = parser.parse_args()
    
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        # Auto-discovery
        # Check both local 'checkpoints' and 'runs' directories
        candidates = []
        
        for d in ["checkpoints", "runs"]:
             p = Path(d)
             if p.exists():
                 candidates.extend(list(p.rglob("*.pth")))
                 candidates.extend(list(p.rglob("*.pt")))
        
        if not candidates:
            print("❌ No checkpoint provided and no .pth/.pt files found in 'checkpoints/' or 'runs/'")
            sys.exit(1)
            
        # Prioritize 'best_model.pt'
        best_candidates = [c for c in candidates if "best" in c.name.lower()]
        
        if best_candidates:
             # Sort best candidates by modification time
             checkpoint_path = sorted(best_candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
             print(f"🏆 Auto-discovered best model: {checkpoint_path}")
        else:
             # Sort all candidates by modification time
              checkpoint_path = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
              print(f"🔄 Auto-discovered latest checkpoint (no 'best' found): {checkpoint_path}")
    
    # Handle fraction parameter
    limit_samples = args.limit_samples
    if args.fraction is not None:
        if args.limit_samples is not None:
            print("⚠️  Both --fraction and --limit-samples provided. Using --limit-samples.")
        else:
            # Will calculate actual limit after dataset is loaded
            limit_samples = args.fraction  # Pass as float, will be converted in run_evaluation
    
    run_evaluation(
        Path(checkpoint_path),
        Path(args.dataset_root),
        args.effect,
        Path("evaluation_outputs"),
        limit_samples=limit_samples,
        save_preds=args.save_preds,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
