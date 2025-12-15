from typing import Dict, List, Any
import torch
import numpy as np
from .metrics import calculate_all_metrics

class MetricAnalyzer:
    """
    Computes and aggregates metrics for audio predictions.
    """
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.results = [] # List of dicts

    def process_batch(self, batch_result: Dict[str, torch.Tensor]):
        """
        Calculate metrics for a batch of predictions and store them.
        Args:
            batch_result: Dict with 'input', 'target', 'prediction' (tensors)
        """
        preds = batch_result['prediction']
        targets = batch_result['target']
        
        batch_size = preds.shape[0]
        for i in range(batch_size):
            p = preds[i]
            t = targets[i]
            
            # Calculate metrics using the existing metrics library
            m = calculate_all_metrics(p, t, self.sample_rate)
            self.results.append(m)

    def get_aggregated_results(self) -> Dict[str, Any]:
        """
        Aggregate stored metrics into mean/std stats.
        """
        if not self.results:
            return {}
            
        metric_names = list(self.results[0].keys())
        aggregated = {}
        
        for name in metric_names:
            values = [r[name] for r in self.results]
            aggregated[name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
            
        return {
            'metrics': aggregated,
            'per_sample_metrics': self.results
        }
    
    def generate_summary(self) -> str:
        """
        Generate a text summary of the aggregated results.
        """
        agg = self.get_aggregated_results().get('metrics', {})
        if not agg:
            return "No metrics computed."
            
        lines = ["=== Evaluation Summary ==="]
        for metric, stats in agg.items():
            lines.append(f"{metric:<30}: Mean={stats['mean']:.4f} Std={stats['std']:.4f}")
        return "\n".join(lines)
