import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional
import numpy as np

class GuitarDataset(Dataset):
    def __init__(self, input_data: torch.Tensor, target_data: torch.Tensor, block_size: int = 512):
        self.block_size = block_size
        self.input_data = input_data
        self.target_data = target_data
        
        # Ensure data is reshaped to (num_blocks, block_size, channels) or similar
        # For now assuming (N, 1) or (N,) and we chunk it
        self._prepare_data()

    def _prepare_data(self):
        # Simple reshaping logic - can be expanded
        # If data is 1D or 2D (samples, channels), chunk it into blocks
        if self.input_data.dim() == 1:
            self.input_data = self.input_data.unsqueeze(-1)
        if self.target_data.dim() == 1:
            self.target_data = self.target_data.unsqueeze(-1)
            
        num_samples = self.input_data.shape[0]
        num_blocks = num_samples // self.block_size
        
        # Truncate to full blocks
        self.input_data = self.input_data[:num_blocks * self.block_size]
        self.target_data = self.target_data[:num_blocks * self.block_size]
        
        # Reshape to (num_blocks, block_size, channels)
        # Assuming channels is last dim
        self.input_data = self.input_data.view(num_blocks, self.block_size, -1)
        self.target_data = self.target_data.view(num_blocks, self.block_size, -1)
        
        # Transpose to (num_blocks, channels, block_size) if needed by model
        # Usually models expect (batch, channels, time) or (batch, time, channels)
        # Let's stick to (batch, time, channels) for now as per common audio RNNs, 
        # but PyTorch Conv1d expects (batch, channels, time).
        # Let's use (batch, channels, time) for compatibility with Conv1d
        self.input_data = self.input_data.transpose(1, 2)
        self.target_data = self.target_data.transpose(1, 2)

    def __len__(self) -> int:
        return self.input_data.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input_data[idx], self.target_data[idx]
