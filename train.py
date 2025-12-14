import argparse
import torch
import os
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
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="runs", help="Output directory for checkpoints")
    parser.add_argument("--lr", type=float, default=0.00001, help="Learning rate")
    parser.add_argument("--metrics_out", type=str, default=None, help="Path to save metrics JSON")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    args = parser.parse_args()

    # 1. Load Configuration
    config = ProjectConfig()
    
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
            # Override defaults
            if 'device' in yaml_config:
                # device is handled locally in script, not in config object usually, but let's see
                pass
            if 'batch_size' in yaml_config:
                config.training.batch_size = yaml_config['batch_size']
            if 'epochs' in yaml_config:
                config.training.epochs = yaml_config['epochs']
            if 'data_root' in yaml_config:
                args.dataset_root = yaml_config['data_root']
            if 'metrics_out' in yaml_config:
                args.metrics_out = yaml_config['metrics_out']
            if 'checkpoint_dir' in yaml_config:
                config.output_dir = yaml_config['checkpoint_dir']
                
    # CLI args override config file
    if args.model: config.model.name = args.model
    if args.epochs != 100: config.training.epochs = args.epochs # Only override if changed from default? Or always?
    # Let's say CLI args always take precedence if provided explicitly.
    # But argparse defaults make it hard to know if user provided it.
    # For now, let's assume if config file is passed, we trust it, but CLI args override.
    
    config.training.batch_size = args.batch_size
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
    from src.training.loss import MSELoss, ESRLoss, CombinedLoss, MultiScaleSpectralLoss
    
    loss_fn = CombinedLoss({
        MSELoss(): 1.0,
        ESRLoss(): 0.5
    })
    
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
    
    # Initialize Logger
    logger = create_kaggle_logger(
        project="forger-nfx",
        run_name=f"{args.target_folder}_{config.model.name}",
        config=config.model_dump()
    )

    metrics = {
        "history": [],
        "final_loss": None,
        "best_val_loss": float('inf'),
        "config": config.model_dump()
    }
    
    run_dir = paths.get_run_path(f"{args.target_folder}_{config.model.name}")
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Select fixed samples for tracking (first 3)
    fixed_val_samples = []
    try:
        inputs, targets = next(iter(val_loader))
        num_idx = min(3, inputs.shape[0])
        for i in range(num_idx):
            fixed_val_samples.append((inputs[i:i+1].to(device), targets[i:i+1].to(device)))
    except StopIteration:
        print("⚠️ Warning: Validation loader is empty, cannot pick fixed samples.")

    import wandb # Import for explicit type checking if needed, though logger handles it
    
    def checkpoint_callback(epoch, train_loss, val_loss, val_metrics=None):
        # Log to W&B / Local
        log_dict = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch": epoch + 1
        }
        if val_metrics:
            for k, v in val_metrics.items():
                log_dict[f"val/{k}"] = v
        
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
                        pred = model(inp)
                        
                        # Prepare audio for W&B
                        # wandb.Audio expects numpy array
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
            "train_loss": train_loss,
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
    
    # --- POST-TRAINING EVALUATION ---
    print("\n📊 Starting Post-Training Evaluation...")
    
    # Load Best Model
    best_path = run_dir / "best_model.pt"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded best model from epoch {checkpoint.get('epoch', '?')} (Val Loss: {checkpoint.get('loss', '?'):.6f})")
    else:
        print("⚠️ Best model not found, using final model state.")

    from src.evaluation.evaluator import ModelEvaluator
    from src.evaluation.visualizer import MetricsVisualizer
    
    # Run Evaluation
    evaluator = ModelEvaluator(model, test_loader, device=device, sample_rate=config.audio.sample_rate)
    eval_results = evaluator.evaluate(save_predictions=True, output_dir=run_dir / "predictions")
    
    # Save JSON Results
    evaluator.save_results(eval_results, run_dir / "evaluation_results.json", format='json')
    
    # Generate HTML Report
    visualizer = MetricsVisualizer(eval_results, sample_rate=config.audio.sample_rate)
    report_path = run_dir / "evaluation_report.html"
    
    # Load generic history for plot if available
    train_history_plot = None
    if metrics["history"]:
        train_history_plot = metrics["history"]
        
    visualizer.create_full_report(report_path, training_history=train_history_plot)
    
    # Log to W&B
    if logger.use_wandb:
        try:
            import wandb
            print("🚀 Logging Evaluation Artifacts to W&B...")
            
            # 1. Log Aggregated Metrics
            wandb_metrics = {}
            for k, v in eval_results['metrics'].items():
                if isinstance(v, dict) and 'mean' in v:
                    wandb_metrics[f"test/{k}"] = v['mean']
            wandb.log(wandb_metrics)
            
            # 2. Log HTML Report
            if report_path.exists():
                wandb.log({"evaluation_report": wandb.Html(open(report_path, encoding='utf-8').read())})
            
            # 3. Log Best Model Artifact
            if best_path.exists():
                artifact = wandb.Artifact(name=f"{logger.project}_{logger.run_name}_best", type="model")
                artifact.add_file(str(best_path))
                wandb.log_artifact(artifact)
                
            # 4. Log Audio Samples & Spectrograms (Top 3)
            # Create a W&B Table
            columns = ["id", "input_audio", "target_audio", "pred_audio", "spectrogram_comparison", "freq_response"]
            table = wandb.Table(columns=columns)
            
            # Get some samples from test loader for logging
            # Re-run inference on a few samples specifically for logging images
            model.eval()
            inputs, targets = next(iter(test_loader))
            inputs = inputs.to(device)
            preds = model(inputs)
            
            num_samples = min(3, inputs.shape[0]) # Requested 3 samples
            for i in range(num_samples):
                inp = inputs[i].detach()
                tgt = targets[i].detach()
                prd = preds[i].detach()
                
                # Audio
                sr = config.audio.sample_rate
                wb_inp = wandb.Audio(inp.cpu().numpy().flatten(), sample_rate=sr, caption="Input")
                wb_tgt = wandb.Audio(tgt.cpu().numpy().flatten(), sample_rate=sr, caption="Target")
                wb_prd = wandb.Audio(prd.cpu().numpy().flatten(), sample_rate=sr, caption="Predicted")
                
                # Plots (using Visualizer helper)
                fig_spec = visualizer.plot_spectrogram_comparison(prd, tgt)
                img_spec = wandb.Image(fig_spec) # Plotly figure to Image
                
                fig_freq = visualizer.plot_frequency_response(prd, tgt)
                img_freq = wandb.Image(fig_freq)
                
                table.add_data(i, wb_inp, wb_tgt, wb_prd, img_spec, img_freq)
                
            wandb.log({"evaluation_samples": table})
            print("✅ Logged evaluation samples to W&B")
            
        except Exception as e:
            print(f"⚠️ Failed to log to W&B: {e}")

    # Export metrics (Legacy)
    if args.metrics_out:
        import json
        metrics["final_loss"] = metrics["history"][-1]["val_loss"] if metrics["history"] else 0.0
        with open(args.metrics_out, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {args.metrics_out}")

if __name__ == "__main__":
    main()
