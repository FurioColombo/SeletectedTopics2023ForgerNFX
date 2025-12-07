import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.training.trainer import Trainer
from src.training.loss import MSELoss
from src.models.lstm import LSTMModel
from src.config.config import TrainingConfig, ModelConfig

def test_training_step():
    # Setup
    model_config = ModelConfig(name="lstm", hidden_size=8)
    model = LSTMModel(model_config)
    
    train_config = TrainingConfig(epochs=1, batch_size=2)
    
    # Dummy data
    x = torch.randn(10, 1, 100)
    y = torch.randn(10, 1, 100)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=2)
    
    loss_fn = MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        config=train_config,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device="cpu"
    )
    
    # Run one epoch
    loss = trainer.train_epoch()
    assert isinstance(loss, float)
    assert loss > 0

def test_loss_shapes():
    loss_fn = MSELoss()
    pred = torch.randn(2, 1, 100)
    target = torch.randn(2, 1, 100)
    loss = loss_fn(pred, target)
    assert loss.ndim == 0
