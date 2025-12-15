from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pathlib import Path

class BaseLogger(ABC):
    """
    Abstract base class for all loggers (WandB, Local, Kaggle, etc.).
    Defines the contract for logging training, validation, and evaluation results.
    """
    
    @abstractmethod
    def log_training(self, epoch: int, batch: int, loss: float, lr: float):
        """Log training metrics for a specific batch/step."""
        pass
    
    @abstractmethod
    def log_validation(self, epoch: int, avg_loss: float, metrics: Dict[str, float]):
        """Log aggregated validation metrics for an epoch."""
        pass
    
    @abstractmethod
    def log_test_quantitative(self, results: Dict[str, Any]):
        """Log quantitative evaluation results (e.g., from SegmentRunner)."""
        pass
    
    @abstractmethod
    def log_test_qualitative(self, sequences: List[Dict[str, Any]], visualizer=None):
        """
        Log qualitative evaluation results (audio, plots).
        Args:
            sequences: List of results from SequenceRunner (input, target, pred, etc.)
            visualizer: Optional MetricsVisualizer instance for generating plots.
        """
        pass
        
    @abstractmethod
    def finish(self):
        """Clean up and close the logger."""
        pass
