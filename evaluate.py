import argparse
import torch
from pathlib import Path
import yaml
import wandb
import sys
from datetime import datetime

from src.models.lstm import LSTMModel
from src.models.conv import ConvModel
from src.data.egfx import EGFxDataset
from src.inference.runners import SegmentRunner, SequenceRunner
from src.evaluation.analyzer import MetricAnalyzer
from src.evaluation.visualizer import MetricsVisualizer

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
    
    # 1. Load Model
    print("Loading model...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Infer architecture
    if 'conv' in str(checkpoint_path).lower():
         model = ConvModel() # Default/Placeholder logic
    else:
        # Robust load for LSTM
        state_dict = checkpoint.get('state_dict', checkpoint)
        hidden_size = 16 # fallback
        if 'lstm.weight_hh_l0' in state_dict:
            hidden_size = state_dict['lstm.weight_hh_l0'].shape[1]
        model = LSTMModel(hidden_size=hidden_size)
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # 2. Setup Data
    input_root = dataset_root / effect_name / "Train" # Use Train for now or Test if split
    # Note: Kaggle dataset structure might differ. Assuming standard EGFx structure.
    # If explicit paths needed, user can adjust.
    # Actually, let's use the provided root.
    
    # 3. PHASE 1: Quantitative (Segments)
    print("\n📊 Phase 1: Quantitative Evaluation (Segments)")
    dataset = EGFxDataset(
        input_root=str(dataset_root / effect_name / "Train"), # Or Test
        output_root=str(dataset_root / effect_name / "Target"),
        block_size=2048
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=2)
    
    analyzer = MetricAnalyzer()
    seg_runner = SegmentRunner(model, dataloader, device, limit_batches=(limit_samples//32 if limit_samples else None))
    
    for batch_res in seg_runner.run():
        analyzer.process_batch(batch_res)
        
    quantitative_results = analyzer.get_aggregated_results()
    print(analyzer.generate_summary())
    
    # 4. PHASE 2: Qualitative (Full Sequences)
    print("\n👂 Phase 2: Qualitative Evaluation (Full Sequences)")
    # Find a few test files
    train_dir = dataset_root / effect_name / "Train"
    target_dir = dataset_root / effect_name / "Target"
    all_files = list(train_dir.glob("*.wav"))
    test_files = all_files[:3] # process first 3 files fully
    
    file_pairs = [(f, target_dir / f.name) for f in test_files if (target_dir / f.name).exists()]
    
    # ... (Imports)
    from src.logging.wandb_logger import WandbLogger
    
    # ... (Setup)
    
    # Init Logger
    logger = WandbLogger(
        project="forger-nfx",
        config={"checkpoint": str(checkpoint_path), "effect": effect_name},
        name=f"eval_{effect_name}_{datetime.now().strftime('%H%M')}",
        tags=["modular-eval"]
    )
    
    # ... (Phase 1)
    
    quantitative_results = analyzer.get_aggregated_results()
    print(analyzer.generate_summary())
    
    # Log Quantitative
    logger.log_test_quantitative(quantitative_results)
    
    # ... (Phase 2)
    
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
        ckpt_dir = Path("checkpoints")
        if not ckpt_dir.exists():
            print("❌ No checkpoint provided and 'checkpoints' directory not found.")
            sys.exit(1)
        
        ckpts = list(ckpt_dir.glob("*.pth"))
        if not ckpts:
            print("❌ No .pth files found in 'checkpoints' directory.")
            sys.exit(1)
            
        # Sort by modification time (latest first)
        checkpoint_path = sorted(ckpts, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        print(f"🔄 Auto-discovered latest checkpoint: {checkpoint_path}")
    
    run_evaluation(
        Path(checkpoint_path),
        Path(args.dataset_root),
        args.effect,
        Path("evaluation_outputs"),
        limit_samples=args.limit_samples,
        save_preds=args.save_preds,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
