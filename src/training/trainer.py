import torch
from torch.utils.data import DataLoader
from typing import Optional, Callable
from tqdm import tqdm
from src.config.config import TrainingConfig
from src.models.base import BaseAudioModel
from .loss import AudioLoss
from .early_stopping import EarlyStopping
from .schedulers import create_scheduler

class Trainer:
    def __init__(
        self,
        model: BaseAudioModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        loss_fn: AudioLoss,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        validation_loss_fns: Optional[dict] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.validation_loss_fns = validation_loss_fns or {}
        
        # Initialize Scheduler
        self.scheduler = create_scheduler(
            optimizer,
            config.scheduler_type,
            config.scheduler_params,
            initial_lr=config.learning_rate
        )
        
        # Initialize Early Stopping
        self.early_stopping = None
        if config.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=config.early_stopping_patience,
                min_delta=0.0001,
                mode='min'
            )
        
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        
        # Custom printer for Kaggle transparency without spam
        # pbar = tqdm(self.train_loader, desc="Training", leave=False, mininterval=10.0) 
        
        for batch_idx, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            loss.backward()
            self.optimizer.step()
            
            # Step Scheduler (Batch)
            self.scheduler.step_batch(batch_idx)

            total_loss += loss.item()
            
            # Print status every 1000 batches to avoid spam
            if batch_idx % 1000 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  Batch {batch_idx}/{len(self.train_loader)} - Loss: {loss.item():.3f} - LR: {current_lr:.2e}")
            
        return total_loss / len(self.train_loader)
        
    def validate(self) -> tuple[float, dict]:
        self.model.eval()
        total_loss = 0.0
        extra_losses = {name: 0.0 for name in self.validation_loss_fns}
        
        # pbar = tqdm(self.val_loader, desc="Validation", leave=False, mininterval=5.0)
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.val_loader):
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                
                # Main loss
                loss = self.loss_fn(pred, y)
                total_loss += loss.item()
                
                # Extra losses
                for name, fn in self.validation_loss_fns.items():
                    extra_loss = fn(pred, y)
                    extra_losses[name] += extra_loss.item()
                
                # pbar.set_postfix({'loss': f'{loss.item():.6f}'})
                
        avg_loss = total_loss / len(self.val_loader)
        avg_extras = {name: val / len(self.val_loader) for name, val in extra_losses.items()}
        
        return avg_loss, avg_extras
        
    def train(self, callbacks: Optional[list] = None):
        # Use simpler output for Kaggle (no ncols, but explicit file or just simple iteration)
        # Using mininterval helps reduce log spam
        epoch_pbar = tqdm(range(self.config.epochs), desc="Epochs", mininterval=10.0)
        for epoch in epoch_pbar:
            train_loss = self.train_epoch()
            val_loss, val_metrics = self.validate()
            
            # Step Scheduler (Epoch)
            self.scheduler.step_epoch(val_loss)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_pbar.set_postfix({
                'train': f'{train_loss:.4f}',
                'val': f'{val_loss:.4f}',
                'lr': f'{current_lr:.2e}'
            })
            print(f"\nEpoch {epoch+1}/{self.config.epochs} - Train: {train_loss:.6f} - Val: {val_loss:.6f} - LR: {current_lr:.2e}")
            
            if callbacks:
                for cb in callbacks:
                    cb(epoch, train_loss, val_loss, val_metrics)
            
            # Early Stopping Check
            if self.early_stopping:
                if self.early_stopping(val_loss):
                    print(f"\n🛑 Early stopping triggered after {epoch+1} epochs.")
                    break
