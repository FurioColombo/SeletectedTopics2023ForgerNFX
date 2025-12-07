import torch
from src.models.base import BaseAudioModel
from src.utils.io import load_audio, save_audio
import os

def run_inference(
    model: BaseAudioModel,
    input_path: str,
    output_path: str,
    device: str = "cpu"
):
    """
    Run inference on an audio file.
    """
    model.to(device)
    model.eval()
    
    # Load audio
    waveform = load_audio(input_path) # (channels, time)
    
    # Add batch dimension: (1, channels, time)
    input_tensor = waveform.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output_tensor = model(input_tensor)
        
    # Remove batch dimension
    output_waveform = output_tensor.squeeze(0).cpu()
    
    # Save audio
    save_audio(output_path, output_waveform)
    
    return output_waveform
