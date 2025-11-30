import torch
from torch.utils.data import DataLoader
from typing import Optional, Callable
from project.config.config import TrainingConfig
from project.models.base import BaseAudioModel
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
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
        
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss = self.loss_fn(pred, y)
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)
        
    def train(self, callbacks: Optional[list] = None):
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            print(f"Epoch {epoch+1}/{self.config.epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
            
            if callbacks:
                for cb in callbacks:
                    cb(epoch, train_loss, val_loss)
