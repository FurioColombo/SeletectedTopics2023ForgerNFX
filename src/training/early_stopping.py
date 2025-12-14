"""
Early stopping implementation.
"""
from typing import Optional, Dict

class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving.
    """
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: How many epochs to wait after last time validation loss improved.
            min_delta: Minimum change in the monitored quantity to qualify as an improvement.
            mode: one of {'min', 'max'}. In 'min' mode, training will stop when the
                quantity monitored has stopped decreasing; in 'max' mode it will stop
                when the quantity monitored has stopped increasing.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = float('inf') if mode == 'min' else float('-inf')
        
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss/metric
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.mode == 'min':
            score = -val_loss
        else:
            score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_loss = val_loss
        elif score < self.best_score + self.min_delta:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Improved
            self.best_score = score
            self.best_loss = val_loss
            self.counter = 0
            
        return self.early_stop
