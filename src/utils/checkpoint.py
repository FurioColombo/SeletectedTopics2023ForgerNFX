import torch
import os
import json
from typing import Any, Dict
from src.models.base import BaseAudioModel

def save_checkpoint(
    model: BaseAudioModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str
):
    """Save PyTorch checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': model.get_config()
    }
    torch.save(state, path)

def load_checkpoint(path: str, model: BaseAudioModel, optimizer: torch.optim.Optimizer = None) -> Dict[str, Any]:
    """Load PyTorch checkpoint."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
        
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    return checkpoint

def save_for_rtneural(model: BaseAudioModel, path: str):
    """Save model weights as JSON for RTNeural."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    state_dict = model.state_dict()
    # Convert tensors to lists
    json_dict = {k: v.cpu().detach().numpy().tolist() for k, v in state_dict.items()}
    
    with open(path, 'w') as f:
        json.dump(json_dict, f)
