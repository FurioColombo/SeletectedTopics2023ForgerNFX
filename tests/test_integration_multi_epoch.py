import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.training.trainer import Trainer
from src.training.loss import MSELoss
from src.models.lstm import LSTMModel
from src.config.config import TrainingConfig, ModelConfig

def test_multi_epoch_training():
    # Setup
    torch.manual_seed(42)
    model_config = ModelConfig(name="lstm", hidden_size=8)
    model = LSTMModel(model_config)
    
    train_config = TrainingConfig(epochs=5, batch_size=4, learning_rate=0.01)
    
    # Synthetic data: simple sine wave mapping
    # Input: sine wave
    # Target: distorted sine wave (tanh)
    t = torch.linspace(0, 1, 1000)
    x = torch.sin(2 * torch.pi * 10 * t).unsqueeze(0).unsqueeze(0) # (1, 1, 1000)
    # Repeat to make batch
    x = x.repeat(20, 1, 1) # (20, 1, 1000)
    y = torch.tanh(x * 2) # Target distortion
    
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=train_config.batch_size)
    
    loss_fn = MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    
    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        config=train_config,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device="cpu"
    )
    
    # Track losses
    losses = []
    def callback(epoch, train_loss, val_loss):
        losses.append(train_loss)
        
    trainer.train(callbacks=[callback])
    
    # Check if loss decreased
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]} -> {losses[-1]}"
    assert len(losses) == 5
