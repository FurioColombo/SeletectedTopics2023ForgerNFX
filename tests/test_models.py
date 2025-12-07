import pytest
import torch
from src.models.lstm import LSTMModel
from src.models.conv import ConvModel
from src.config.config import ModelConfig

def test_lstm_forward():
    config = ModelConfig(name="lstm", hidden_size=16)
    model = LSTMModel(config)
    
    # Input: (batch, channels, time)
    batch_size = 4
    time_steps = 100
    x = torch.randn(batch_size, 1, time_steps)
    
    y = model(x)
    
    # Output should be same shape
    assert y.shape == x.shape
    
def test_conv_forward():
    config = ModelConfig(name="conv", hidden_size=16, kernel_size=3)
    model = ConvModel(config)
    
    batch_size = 4
    time_steps = 100
    x = torch.randn(batch_size, 1, time_steps)
    
    y = model(x)
    
    assert y.shape == x.shape
