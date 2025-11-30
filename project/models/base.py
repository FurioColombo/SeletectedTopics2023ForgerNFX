import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAudioModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            x: Input tensor of shape (batch, channels, time) or (batch, time, channels)
        Returns:
            Output tensor of same shape as input (usually)
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration for saving."""
        return {}
