import torch
import torchaudio
import os
from typing import Tuple

def load_audio(path: str, target_sr: int = 44100) -> torch.Tensor:
    """
    Load audio file and resample if necessary.
    Returns tensor of shape (channels, time).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
        
    waveform, sr = torchaudio.load(path)
    
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
        
    return waveform

def save_audio(path: str, waveform: torch.Tensor, sr: int = 44100):
    """
    Save audio tensor to file.
    waveform: (channels, time)
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchaudio.save(path, waveform, sr)
