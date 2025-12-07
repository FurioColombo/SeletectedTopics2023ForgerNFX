import torch
import torch.nn as nn
from .base import BaseAudioModel
from src.config.config import ModelConfig

class ConvModel(BaseAudioModel):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.kernel_size = config.kernel_size or 3
        self.hidden_size = config.hidden_size
        
        # Simple Conv1d model
        # Input: (batch, channels, time)
        self.conv1 = nn.Conv1d(1, self.hidden_size, kernel_size=self.kernel_size, padding=self.kernel_size//2)
        self.act1 = nn.Tanh()
        self.conv2 = nn.Conv1d(self.hidden_size, 1, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, time)
        # If input is (batch, time, channels), transpose it
        if x.shape[2] == 1 and x.shape[1] != 1:
             x = x.transpose(1, 2)
             
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        
        return out
        
    def get_config(self):
        return {"name": "conv", "kernel_size": self.kernel_size}
