import torch
import numpy as np

def calculate_esr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate Error Signal Ratio (ESR).
    pred, target: (channels, time) or (batch, channels, time)
    """
    error = target - pred
    numerator = torch.sum(error ** 2)
    denominator = torch.sum(target ** 2) + 1e-8
    return (numerator / denominator).item()

def calculate_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate Mean Squared Error."""
    return torch.mean((pred - target) ** 2).item()
