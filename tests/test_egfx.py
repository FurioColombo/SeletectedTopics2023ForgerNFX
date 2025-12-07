import pytest
import torch
import os
from src.data.egfx import EGFxDataset
from src.utils.io import save_audio

def test_egfx_dataset(tmp_path):
    # Create dummy directory structure
    input_dir = tmp_path / "Clean"
    output_dir = tmp_path / "Distorted"
    
    # Create subdirectories
    (input_dir / "Bridge").mkdir(parents=True)
    (output_dir / "Bridge").mkdir(parents=True)
    
    # Create dummy audio files
    sr = 44100
    # 2 seconds of audio
    waveform = torch.randn(1, sr * 2)
    
    # Create matching files in subdirectory
    save_audio(str(input_dir / "Bridge" / "file1.wav"), waveform, sr)
    save_audio(str(output_dir / "Bridge" / "file1.wav"), waveform, sr)
    
    # Create non-matching files
    save_audio(str(input_dir / "file2.wav"), waveform, sr)
    save_audio(str(output_dir / "file3.wav"), waveform, sr)
    
    # Test dataset
    dataset = EGFxDataset(
        input_root=str(input_dir),
        output_root=str(output_dir),
        block_size=512,
        sample_rate=sr
    )
    
    # Should find 1 pair
    assert len(dataset.input_files) == 1
    
    # Check data loading
    # Total samples: 44100 * 2 = 88200
    # Block size: 512
    # Num blocks: 88200 // 512 = 172
    assert len(dataset) == 172
    
    x, y = dataset[0]
    assert x.shape == (1, 512)
    assert y.shape == (1, 512)
