import warnings
import os

# Suppress pydantic warnings BEFORE imports
warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import argparse
import torch
from pathlib import Path
import sys

# Ensure src module is importable
repo_root = Path(__file__).parent
sys.path.append(str(repo_root))


from src.config.config import ProjectConfig
from src.config.paths import paths
from src.data.egfx import EGFxDataset
from src.data.loader import create_dataloaders
from src.models.lstm import LSTMModel
from src.models.conv import ConvModel
from src.training.trainer import Trainer
from src.training.loss import MSELoss, ESRLoss, CombinedLoss
from src.monitoring.logger import create_kaggle_logger

def main():
    parser = argparse.ArgumentParser(description="Train a guitar audio distortion model")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root directory of the dataset (e.g. EGFxDataset)")
    parser.add_argument("--input_folder", type=str, default="Clean", help="Name of input folder (e.g. Clean)")
    parser.add_argument("--target_folder", type=str, required=True, help="Name of target folder (e.g. TubeScreamer)")
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "conv"], help="Model type")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="runs", help="Output directory for checkpoints")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--metrics_out", type=str, default=None, help="Path to save metrics JSON")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    args = parser.parse_args()

    # 1. Load Configuration
    from src.config.config import load_config
    config = load_config(args.config)
    
    # CLI args override config file ONLY if provided
    if args.model: config.model.name = args.model
    if args.epochs is not None: config.training.epochs = args.epochs
    
    if args.dataset_root: args.dataset_root = args.dataset_root # Ensure this is passed if needed, mainly for data loader
    
    # Force CLI overrides ONLY if provided (not default)
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.output_dir != "runs": config.output_dir = args.output_dir
    
    print(f"Configuration loaded: Model={config.model.name}, Epochs={config.training.epochs}")

    # 2. Load Data
    input_root = os.path.join(args.dataset_root, args.input_folder)
    output_root = os.path.join(args.dataset_root, args.target_folder)
    
    print(f"Loading data from:\n  Input: {input_root}\n  Target: {output_root}")
    
    try:
        dataset = EGFxDataset(
            input_root=input_root,
            output_root=output_root,
            block_size=config.audio.block_size,
            sample_rate=config.audio.sample_rate
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    train_loader, val_loader, test_loader = create_dataloaders(dataset, config)
    
    print(f"Dataset created. Train batches: {len(train_loader)}")

    # 3. Initialize Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if config.model.name == "lstm":
        model = LSTMModel(config.model)
    elif config.model.name == "conv":
        model = ConvModel(config.model)
    else:
        raise ValueError(f"Unknown model: {config.model.name}")
        
    model = model.to(device)

    # 4. Setup Training
    # 4. Setup Training
    # Loss: Combination of MSE and ESR
    # Loss: Combination of MSE and ESR
    from src.training.loss import MSELoss, ESRLoss, CombinedLoss, MultiScaleSpectralLoss
    
    # Determine Loss Weights
    # Defaults
    mse_weight = 100.0
    esr_weight = 0.5
    spectral_weight = 0.0 # Default off unless in config
    
    # distinct loading logic to handle potential missing Pydantic fields
    loss_weights_found = False
    
    # 1. Try Config Object
    if hasattr(config.training, 'loss_weights') and config.training.loss_weights:
        w = config.training.loss_weights
        if isinstance(w, dict):
            mse_weight = float(w.get('mse', mse_weight))
            esr_weight = float(w.get('esr', esr_weight))
            spectral_weight = float(w.get('spectral', spectral_weight))
            loss_weights_found = True
            
    # 2. Key Fallback: functions if Pydantic model ignores 'loss_weights'
    if not loss_weights_found and args.config:
        import yaml
        try:
            with open(args.config, 'r') as f:
                raw_conf = yaml.safe_load(f)
                if 'loss_weights' in raw_conf:
                    mse_weight = float(raw_conf['loss_weights'].get('mse', mse_weight))
                    esr_weight = float(raw_conf['loss_weights'].get('esr', esr_weight))
                    spectral_weight = float(raw_conf['loss_weights'].get('spectral', spectral_weight))
                    print(f"Loaded loss weights from YAML: MSE={mse_weight}, ESR={esr_weight}, Spectral={spectral_weight}")
        except Exception as e:
            print(f"Warning: Failed to parse raw config for loss weights: {e}")

    print(f"Training with Loss Weights -> MSE: {mse_weight}, ESR: {esr_weight}, Spectral: {spectral_weight}")

    # Build Loss Ensemble
    losses = {
        MSELoss(): mse_weight,
        ESRLoss(): esr_weight
    }
    
    if spectral_weight > 0:
        # Note: MultiScaleSpectralLoss uses FFT logic which may need specific device handling
        # It's an nn.Module, so Trainer keeps it on device.
        losses[MultiScaleSpectralLoss()] = spectral_weight

    loss_fn = CombinedLoss(losses)
    
    # Secondary validation metrics
    validation_loss_fns = {
        "spectral_loss": MultiScaleSpectralLoss()
    }
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.training,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        validation_loss_fns=validation_loss_fns
    )

    # 5. Run Training
    print("Starting training...")
    
    # Generate run name with date
    from datetime import datetime
    date_str = datetime.now().strftime("%y%m%d")
    run_name = f"{date_str}_{args.target_folder}_{config.model.name}"
    
    # Initialize Logger
    logger = create_kaggle_logger(
        project="forger-nfx",
        run_name=run_name,
        config=config.model_dump()
    )

    metrics = {
        "history": [],
        "final_loss": None,
        "best_val_loss": float('inf'),
        "config": config.model_dump()
    }
    
    run_dir = paths.get_run_path(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Select fixed samples for tracking (full files)
    fixed_val_samples = []
    try:
        # Manually load a few files from the validation set (or just dataset root)
        # We need full length audio for listening, not 512-sample blocks
        from src.utils.io import load_audio
        import random
        
        # Get list of files in input_root
        input_files = sorted([f for f in os.listdir(input_root) if f.endswith('.wav')])
        if len(input_files) > 0:
            # Pick 3 random files
            selected_files = input_files[:3] if len(input_files) <= 3 else random.sample(input_files, 3)
            
            for fname in selected_files:
                inp_path = os.path.join(input_root, fname)
                tgt_path = os.path.join(output_root, fname)
                
                if os.path.exists(tgt_path):
                    # Load full audio
                    inp_wave = load_audio(inp_path)
                    tgt_wave = load_audio(tgt_path)
                    
                    # Trim to reasonable length for logging (e.g. 5 seconds max)
                    max_len = config.audio.sample_rate * 5
                    if inp_wave.shape[-1] > max_len:
                        inp_wave = inp_wave[..., :max_len]
                        tgt_wave = tgt_wave[..., :max_len]
                        
                    fixed_val_samples.append((inp_wave.to(device), tgt_wave.to(device)))
                    print(f"Loaded fixed sample: {fname} ({inp_wave.shape[-1]/config.audio.sample_rate:.2f}s)")
        else:
            print("⚠️ No WAV files found in input root for logging.")
            
    except Exception as e:
        print(f"⚠️ Warning: Failed to load fixed samples: {e}")

    import wandb # Import for explicit type checking if needed, though logger handles it
    
    def checkpoint_callback(epoch, train_metrics, val_loss, val_metrics=None):
        # Log to W&B / Local
        current_lr = optimizer.param_groups[0]['lr']
        
        # Handle dict vs float (backward compatibility)
        if isinstance(train_metrics, dict):
            t_loss = train_metrics['combined_loss']
        else:
            t_loss = train_metrics
            train_metrics = {'combined_loss': t_loss}

        # Organized Logging Structure
        log_dict = {
            "run/epoch": epoch + 1,
            "run/learning_rate": current_lr,
            "val/combined_loss": val_loss, # Main validation loss (renamed for clarity)
        }
        
        # Log all training component losses
        for k, v in train_metrics.items():
            # e.g. train/combined_loss, train/MSELoss, train/ESRLoss
            log_dict[f"train/{k}"] = v

        if val_metrics:
            for k, v in val_metrics.items():
                # Ensure val prefix
                key = k if k.startswith("val/") else f"val/{k}"
                log_dict[key] = v
        
        # Log Audio Samples during Checkpoint
        if logger.use_wandb and logger.run and fixed_val_samples:
            model.eval()
            try:
                # Log 3 fixed samples
                sample_rate = config.audio.sample_rate
                
                # Combine into a single W&B Table for cleaner UI or just log individual Audios
                # For checkpoint tracking, individual audio files or a small table is fine.
                # Let's use a small table for each checkpoint to keep it organized
                columns = ["id", "input", "target", "prediction"]
                table = wandb.Table(columns=columns)
                
                with torch.no_grad():
                    for idx, (inp, tgt) in enumerate(fixed_val_samples):
                        # inp is (channels, time), needs (1, channels, time)
                        if inp.dim() == 2:
                            inp_batch = inp.unsqueeze(0)
                        else:
                            inp_batch = inp
                            
                        # Run inference
                        # Note: LSTM model handles arbitrary length if fully convolutional or carefully implemented
                        # LSTMModel: expects (batch, channels, time) -> (batch, time, channels) -> LSTM -> ...
                        # It should handle variable length fine as long as block_size isn't hardcoded in forward (it isn't)
                        pred_batch = model(inp_batch)
                        
                        # Remove batch dim: (1, channels, time) -> (channels, time)
                        pred = pred_batch.squeeze(0)
                        
                        # Prepare audio for W&B
                        # wandb.Audio expects numpy array 1D or (time, channels)
                        # We have (channels, time). Flatten works for mono.
                        wb_inp = wandb.Audio(inp.cpu().numpy().flatten(), sample_rate=sample_rate, caption=f"Input {idx}")
                        wb_tgt = wandb.Audio(tgt.cpu().numpy().flatten(), sample_rate=sample_rate, caption=f"Target {idx}")
                        wb_pred = wandb.Audio(pred.cpu().numpy().flatten(), sample_rate=sample_rate, caption=f"Pred {idx}")
                        
                        table.add_data(idx, wb_inp, wb_tgt, wb_pred)
                
                # Log table to W&B
                logger.run.log({f"val_samples/epoch_{epoch+1}": table}, step=epoch+1)
            except Exception as e:
                print(f"⚠️ Failed to log audio samples: {e}")
            model.train() # Switch back to train mode

        logger.log_metrics(log_dict, step=epoch + 1)
        
        # Keep metrics for backward compatibility with kaggle_train.py
        metrics["history"].append({
            "epoch": epoch + 1,
            "train_loss": t_loss,
            "val_loss": val_loss,
            "extra": val_metrics
        })

        from src.utils.checkpoint import save_checkpoint
        
        # Periodic Checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_path = str(run_dir / f"checkpoint_epoch_{epoch+1}.pt")
            save_checkpoint(model, optimizer, epoch, val_loss, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")
            
        # Best Model Checkpoint
        if val_loss < metrics["best_val_loss"]:
            metrics["best_val_loss"] = val_loss
            best_path = str(run_dir / "best_model.pt")
            save_checkpoint(model, optimizer, epoch, val_loss, best_path)
            print(f"🔥 New best model (Val Loss: {val_loss:.6f}) saved to {best_path}")

    trainer.train(callbacks=[checkpoint_callback])
    
    # Save final model
    final_path = str(run_dir / "final_model.pt")
    from src.utils.checkpoint import save_checkpoint
    save_checkpoint(model, optimizer, config.training.epochs, 0.0, final_path)
    print(f"Training complete! Saved to {final_path}")
    
    # Evaluation is now decoupled. Run evaluate.py for detailed analysis.


    # Export metrics (Legacy)
    if args.metrics_out:
        import json
        metrics["final_loss"] = metrics["history"][-1]["val_loss"] if metrics["history"] else 0.0
        with open(args.metrics_out, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {args.metrics_out}")
        
    print("🏁 Training and evaluation complete. Finishing logger...")
    logger.finish()

if __name__ == "__main__":
    main()
