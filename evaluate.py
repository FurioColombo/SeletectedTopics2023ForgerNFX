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
    limit_samples: int = None,
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

    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=optimal_batch_size, 
        num_workers=2,
        pin_memory=(device == "cuda")
    )
    
    analyzer = MetricAnalyzer()
    seg_runner = SegmentRunner(model, dataloader, device, limit_batches=(limit_samples//32 if limit_samples else None))
    
    for batch_res in seg_runner.run():
        analyzer.process_batch(batch_res)
        
    quantitative_results = analyzer.get_aggregated_results()
    print(analyzer.generate_summary())
    
    # Log Quantitative
    logger.log_test_quantitative(quantitative_results)
    
    # 4. PHASE 2: Qualitative (Full Sequences)
    print("\n👂 Phase 2: Qualitative Evaluation (Full Sequences)")
    # Find a few test files
    all_files = list(input_path.glob("*.wav"))
    test_files = all_files[:3] # process first 3 files fully
    
    file_pairs = [(f, target_path / f.name) for f in test_files if (target_path / f.name).exists()]
    
    seq_runner = SequenceRunner(model, file_pairs, device=device)
    visualizer = MetricsVisualizer(quantitative_results) 
    
    # Collect all qualitative results first
    qualitative_sequences = []
    
    visualizer = MetricsVisualizer(quantitative_results)
    
    for seq_res in seq_runner.run():
        name = seq_res['name']
        print(f"  Processing {name}...")
        
        # Save WAVs locally
        if save_preds:
            (output_dir / "predictions").mkdir(exist_ok=True, parents=True)
            import torchaudio
            torchaudio.save(output_dir / "predictions" / f"{name}_pred.wav", seq_res['prediction'].unsqueeze(0), seq_res['sample_rate'])
        
        qualitative_sequences.append(seq_res)
        
    # Log Qualitative (Batch upload to Table)
    logger.log_test_qualitative(qualitative_sequences, visualizer=visualizer)
    
    logger.finish()
    print("\n✅ Evaluation Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=False, help="Path to model checkpoint. Auto-discovered if not provided.")
    parser.add_argument("--effect", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--save-preds", action="store_true")
    parser.add_argument("--limit-samples", type=int, default=100)
    
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
    
    run_evaluation(
        Path(checkpoint_path),
        Path(args.dataset_root),
        args.effect,
        Path("evaluation_outputs"),
        limit_samples=args.limit_samples,
        save_preds=args.save_preds,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
