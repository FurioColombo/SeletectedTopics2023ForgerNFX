import torch
from torch.utils.data import DataLoader
from typing import Optional, Callable
from tqdm import tqdm
from src.config.config import TrainingConfig
from src.models.base import BaseAudioModel
from .loss import AudioLoss

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
        
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
            
        return total_loss / len(self.train_loader)
        
    def validate(self) -> tuple[float, dict]:
        self.model.eval()
        total_loss = 0.0
        extra_losses = {name: 0.0 for name in self.validation_loss_fns}
        
        pbar = tqdm(self.val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                
                # Main loss
                loss = self.loss_fn(pred, y)
                total_loss += loss.item()
                
                # Extra losses
                for name, fn in self.validation_loss_fns.items():
                    extra_loss = fn(pred, y)
                    extra_losses[name] += extra_loss.item()
                
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})
                
        avg_loss = total_loss / len(self.val_loader)
        avg_extras = {name: val / len(self.val_loader) for name, val in extra_losses.items()}
        
        return avg_loss, avg_extras
        
    def train(self, callbacks: Optional[list] = None):
        epoch_pbar = tqdm(range(self.config.epochs), desc="Epochs")
        for epoch in epoch_pbar:
            train_loss = self.train_epoch()
            val_loss, val_metrics = self.validate()
            
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.6f}',
                'val_loss': f'{val_loss:.6f}'
            })
            print(f"\nEpoch {epoch+1}/{self.config.epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
            
            if callbacks:
                for cb in callbacks:
                    cb(epoch, train_loss, val_loss, val_metrics)
