import pytest
import torch
from src.data.dataset import GuitarDataset
from src.data.loader import create_dataloaders
from src.config.config import ProjectConfig

def test_dataset_shape():
    # Create dummy audio data: 10 seconds at 44.1kHz
    data_len = 44100 * 10
    input_data = torch.randn(data_len)
    target_data = torch.randn(data_len)
    
    block_size = 512
    dataset = GuitarDataset(input_data, target_data, block_size=block_size)
    
    # Check length
    expected_blocks = data_len // block_size
    assert len(dataset) == expected_blocks
    
    # Check item shape: (channels, block_size)
    x, y = dataset[0]
    assert x.shape == (1, block_size)
    assert y.shape == (1, block_size)

def test_dataloaders():
    data_len = 44100 * 1
    input_data = torch.randn(data_len)
    target_data = torch.randn(data_len)
    dataset = GuitarDataset(input_data, target_data, block_size=512)
    
    config = ProjectConfig()
    config.training.batch_size = 10
    
    train_loader, val_loader, test_loader = create_dataloaders(dataset, config)
    
    # Check if loaders yield batches
    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.shape[0] == 10
    assert batch_x.shape[2] == 512
