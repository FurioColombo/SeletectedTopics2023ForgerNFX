"""
Learning Rate Schedulers.
"""
from typing import Dict, Any, Optional
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler

class BaseLRScheduler:
    """Base interface for custom schedulers."""
    def step_batch(self, batch_idx: int):
        """Called after each batch."""
        pass
        
    def step_epoch(self, metrics: Optional[float] = None):
        """Called after each epoch."""
        pass
        
    def state_dict(self) -> Dict:
        return {}
        
    def load_state_dict(self, state_dict: Dict):
        pass

class WarmupReduceLROnPlateau(BaseLRScheduler):
    """
    Scheduler with initial linear warmup followed by ReduceLROnPlateau.
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 100,
        initial_lr: float = 0.001,
        mode: str = 'min',
        factor: float = 0.5,
        patience: int = 5,
        min_lr: float = 1e-6,
        verbose: bool = True
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.current_step = 0
        self.in_warmup = warmup_steps > 0
        
        # Plateau scheduler
        self.plateau_scheduler = ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr
        )
        
    def step_batch(self, batch_idx: int = None):
        """
        Handle warmup during batch steps.
        Increments internal step counter.
        """
        if self.in_warmup:
            self.current_step += 1
            # Linear warmup
            # Even if we exceed warmup_steps slightly (shouldn't happen with logic below), cap at 1.0
            scale = min(1.0, self.current_step / self.warmup_steps)
            lr = self.initial_lr * scale
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
                
            if self.current_step >= self.warmup_steps:
                self.in_warmup = False
                    
    def step_epoch(self, metrics: float):
        """
        Handle plateau reduction at epoch end.
        """
        if not self.in_warmup:
            self.plateau_scheduler.step(metrics)
            
    def state_dict(self) -> Dict:
        return {
            'current_step': self.current_step,
            'in_warmup': self.in_warmup,
            'plateau_state': self.plateau_scheduler.state_dict()
        }
        
    def load_state_dict(self, state_dict: Dict):
        self.current_step = state_dict.get('current_step', 0)
        self.in_warmup = state_dict.get('in_warmup', True)
        if 'plateau_state' in state_dict:
            self.plateau_scheduler.load_state_dict(state_dict['plateau_state'])

def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config_type: str,
    config_params: Dict,
    initial_lr: float
) -> BaseLRScheduler:
    """Factory function to create schedulers."""
    if config_type == "warmup_plateau":
        return WarmupReduceLROnPlateau(
            optimizer,
            initial_lr=initial_lr,
            **config_params
        )
    return BaseLRScheduler() # Dummy fallback
