import argparse
import torch
import os
from pathlib import Path
from project.config.config import ProjectConfig
from project.data.egfx import EGFxDataset
from project.data.loader import create_dataloaders
from project.models.lstm import LSTMModel
from project.models.conv import ConvModel
from project.training.trainer import Trainer
from project.training.loss import MSELoss, ESRLoss, CombinedLoss

def main():
    parser = argparse.ArgumentParser(description="Train a guitar audio distortion model")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root directory of the dataset (e.g. EGFxDataset)")
    parser.add_argument("--input_folder", type=str, default="Clean", help="Name of input folder (e.g. Clean)")
    parser.add_argument("--target_folder", type=str, required=True, help="Name of target folder (e.g. TubeScreamer)")
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "conv"], help="Model type")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="runs", help="Output directory for checkpoints")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
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
    # Loss: Combination of MSE and ESR
    loss_fn = CombinedLoss({
        MSELoss(): 1.0,
        ESRLoss(): 0.5
    })
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.training,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device
    )

    # 5. Run Training
    print("Starting training...")
    
    metrics = {
        "history": [],
        "final_loss": None,
        "config": config.model_dump()
    }
    
    def checkpoint_callback(epoch, train_loss, val_loss):
        metrics["history"].append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss
        })
        
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(config.output_dir, f"{args.target_folder}_{config.model.name}", f"checkpoint_epoch_{epoch+1}.pt")
            from project.utils.checkpoint import save_checkpoint
            save_checkpoint(model, optimizer, epoch, val_loss, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    trainer.train(callbacks=[checkpoint_callback])
    
    # Save final model
    final_path = os.path.join(config.output_dir, f"{args.target_folder}_{config.model.name}", "final_model.pt")
    from project.utils.checkpoint import save_checkpoint
    save_checkpoint(model, optimizer, config.training.epochs, 0.0, final_path)
    print(f"Training complete! Saved to {final_path}")
    
    # Export metrics
    if args.metrics_out:
        import json
        metrics["final_loss"] = metrics["history"][-1]["val_loss"] if metrics["history"] else 0.0
        with open(args.metrics_out, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {args.metrics_out}")

if __name__ == "__main__":
    main()
