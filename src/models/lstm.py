import torch
import torch.nn as nn
from .base import BaseAudioModel
from src.config.config import ModelConfig

class LSTMModel(BaseAudioModel):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_size = 1 # Mono audio
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            batch_first=True
        )
        
        # Output projection
        self.dense = nn.Linear(self.hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, time) -> need (batch, time, channels) for LSTM
        if x.shape[1] == 1:
            x = x.transpose(1, 2).contiguous()
            
        # Debug / Safety check
        if not x.is_contiguous():
            print(f"⚠️ Warning: LSTM input not contiguous! Shape: {x.shape}, Strides: {x.stride()}")
            x = x.contiguous()
            
        lstm_out, _ = self.lstm(x)
        out = self.dense(lstm_out)
        
        # Back to (batch, channels, time)
        out = out.transpose(1, 2)
        return out
        
    def get_config(self):
        return {"name": "lstm", "hidden_size": self.hidden_size}
