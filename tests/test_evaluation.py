import pytest
import torch
import os
from project.evaluation.metrics import calculate_esr, calculate_mse
from project.evaluation.inference import run_inference
from project.models.lstm import LSTMModel
from project.config.config import ModelConfig
from project.utils.io import save_audio

def test_metrics():
    pred = torch.ones(1, 100)
    target = torch.ones(1, 100)
    
    esr = calculate_esr(pred, target)
    assert esr < 1e-6
    
    mse = calculate_mse(pred, target)
    assert mse < 1e-6
    
    # Error case
    pred = torch.zeros(1, 100)
    esr = calculate_esr(pred, target)
    assert esr > 0.9 # Should be 1.0

def test_inference(tmp_path):
    # Create dummy input file
    sr = 44100
    waveform = torch.randn(1, sr)
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    save_audio(str(input_path), waveform, sr)
    
    # Create model
    config = ModelConfig(name="lstm", hidden_size=8)
    model = LSTMModel(config)
    
    # Run inference
    run_inference(model, str(input_path), str(output_path))
    
    assert os.path.exists(output_path)
