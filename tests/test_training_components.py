"""
Tests for training components: EarlyStopping and Schedulers.
"""
import pytest
import torch
import torch.nn as nn
from src.training.early_stopping import EarlyStopping
from src.training.schedulers import WarmupReduceLROnPlateau

class TestEarlyStopping:
    def test_improvement(self):
        es = EarlyStopping(patience=3, mode='min')
        # Improve
        assert not es(1.0)
        assert es.counter == 0
        assert es.best_loss == 1.0
        
        # Improve again
        assert not es(0.9)
        assert es.counter == 0
        assert es.best_loss == 0.9
        
    def test_no_improvement_stops(self):
        es = EarlyStopping(patience=2, mode='min')
        es(1.0) # Best: 1.0
        
        # Worse
        assert not es(1.1)
        assert es.counter == 1
        
        # Worse again -> Stop
        assert es(1.2) # counter=2 >= patience
        assert es.early_stop
        
    def test_min_delta(self):
        es = EarlyStopping(patience=2, min_delta=0.1, mode='min')
        es(1.0)
        
        # 0.95 is improvement < 0.1, so considered no improvement
        assert not es(0.95)
        assert es.counter == 1

class TestScheduler:
    def test_warmup_phase(self):
        model = nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1) # Dummy initial, will be overwritten
        
        initial_lr = 0.1
        warmup_steps = 10
        
        scheduler = WarmupReduceLROnPlateau(
            optimizer, 
            warmup_steps=warmup_steps, 
            initial_lr=initial_lr
        )
        
        # First step: 1/10 of initial_lr
        scheduler.step_batch()
        current_lr = optimizer.param_groups[0]['lr']
        assert abs(current_lr - (initial_lr * 0.1)) < 1e-6
        
        # 5th step: 0.5 of initial_lr
        for _ in range(4): scheduler.step_batch()
        current_lr = optimizer.param_groups[0]['lr']
        assert abs(current_lr - (initial_lr * 0.5)) < 1e-6
        
        # 10th step: Full LR
        for _ in range(5): scheduler.step_batch()
        current_lr = optimizer.param_groups[0]['lr']
        assert abs(current_lr - initial_lr) < 1e-6
        
        assert not scheduler.in_warmup
        
    def test_plateau_phase(self):
        model = nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        
        scheduler = WarmupReduceLROnPlateau(
            optimizer,
            warmup_steps=0, # No warmup
            initial_lr=0.1,
            patience=1,
            factor=0.5
        )
        
        # Initial: 0.1
        scheduler.step_epoch(1.0) # Best: 1.0
        assert optimizer.param_groups[0]['lr'] == 0.1
        
        # Trigger bad epochs
        # Patience=1 means we need >1 bad epochs to reduce? Or 1 bad epoch is tolerated?
        # Typically: 
        # 0: Good (Best)
        # 1: Bad (Counter=1) -> No reduce (1 <= 1)
        # 2: Bad (Counter=2) -> Reduce (2 > 1)
        
        for _ in range(5):
            scheduler.step_epoch(1.0)
            
        assert optimizer.param_groups[0]['lr'] < 0.1 # Should have reduced by now

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
