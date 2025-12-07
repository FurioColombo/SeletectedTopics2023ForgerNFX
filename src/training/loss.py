import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class AudioLoss(nn.Module, ABC):
    @abstractmethod
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pass

class MSELoss(AudioLoss):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred, target)

class ESRLoss(AudioLoss):
    """Error Signal Ratio Loss"""
    def __init__(self):
        super().__init__()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (batch, channels, time)
        error = target - pred
        numerator = torch.sum(error ** 2, dim=2)
        denominator = torch.sum(target ** 2, dim=2) + 1e-8
        return torch.mean(numerator / denominator)

class CombinedLoss(AudioLoss):
    def __init__(self, losses: dict[AudioLoss, float]):
        super().__init__()
        self.losses = losses
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for loss_fn, weight in self.losses.items():
            total_loss += weight * loss_fn(pred, target)
        return total_loss
