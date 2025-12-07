import pytest
import torch
import os
from src.utils.io import save_audio, load_audio
from src.utils.checkpoint import save_checkpoint, load_checkpoint, save_for_rtneural
from src.models.lstm import LSTMModel
from src.config.config import ModelConfig

def test_audio_io(tmp_path):
    # Create dummy audio
    sr = 44100
    waveform = torch.randn(1, sr) # 1 second
    
    path = tmp_path / "test.wav"
    save_audio(str(path), waveform, sr)
    
    assert os.path.exists(path)
    
    loaded = load_audio(str(path), sr)
    assert loaded.shape == waveform.shape

def test_checkpoint(tmp_path):
    config = ModelConfig(name="lstm", hidden_size=8)
    model = LSTMModel(config)
    optimizer = torch.optim.Adam(model.parameters())
    
    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, optimizer, 1, 0.5, str(path))
    
    assert os.path.exists(path)
    
    loaded_ckpt = load_checkpoint(str(path), model, optimizer)
    assert loaded_ckpt['epoch'] == 1
    assert loaded_ckpt['loss'] == 0.5

def test_rtneural_export(tmp_path):
    config = ModelConfig(name="lstm", hidden_size=8)
    model = LSTMModel(config)
    
    path = tmp_path / "model.json"
    save_for_rtneural(model, str(path))
    
    assert os.path.exists(path)
