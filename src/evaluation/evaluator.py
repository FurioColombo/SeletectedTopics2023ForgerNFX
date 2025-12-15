"""
Model evaluator for comprehensive test set evaluation.

Provides batch processing and statistical aggregation of evaluation metrics
for guitar effect models.
"""
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from pathlib import Path
import json
import csv
from tqdm import tqdm

from .metrics import calculate_all_metrics
from ..models.base import BaseAudioModel


class ModelEvaluator:
    """
    Evaluates a trained model on a test dataset with comprehensive metrics.
    """
    
    def __init__(
        self,
        model: BaseAudioModel,
        test_loader: DataLoader,
        device: str = "cpu",
        sample_rate: int = 44100
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model to evaluate
            test_loader: DataLoader for test set
            device: Device to run evaluation on
            sample_rate: Audio sample rate
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.sample_rate = sample_rate
        self.model.eval()
    
    def evaluate(
        self,
        save_predictions: bool = False,
        output_dir: Optional[Path] = None,
        limit_samples: Optional[int] = None
    ) -> Dict:
        """
        Run evaluation on test set.
        
        Args:
            save_predictions: Whether to save predicted audio samples
            output_dir: Directory to save predictions (required if save_predictions=True)
            limit_samples: Optional limit on number of batches to process
        
        Returns:
            Dictionary containing:
                - 'metrics': Dict of aggregated metrics (mean, std, min, max)
                - 'per_sample_metrics': List of metric dicts for each sample
                - 'summary': Text summary of results
        """
        if save_predictions and output_dir is None:
            raise ValueError("output_dir required when save_predictions=True")
        
        if save_predictions:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        all_metrics = []
        audio_samples = []
        
        import torch
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(self.test_loader, desc="Evaluating")):
                # Check limits
                if limit_samples and samples_processed >= limit_samples:
                    break
                    
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Get predictions
                predictions = self.model(inputs)
                
                # Sanity Check for NaNs/Inf in predictions
                if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                    print(f"⚠️ Warning: NaNs or Infs detected in batch {batch_idx}. Replacing with zeros for metrics safety.")
                    predictions = torch.nan_to_num(predictions, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # Calculate metrics for each sample in batch
                batch_size = inputs.shape[0]
                for i in range(batch_size):
                    if limit_samples and samples_processed >= limit_samples:
                        break
                        
                    pred_sample = predictions[i]
                    target_sample = targets[i]
                    input_sample = inputs[i]
                    
                    # Store first 5 samples for W&B logging
                    if len(audio_samples) < 5:
                        audio_samples.append({
                            'idx': batch_idx * batch_size + i,
                            'input': input_sample.cpu(),
                            'target': target_sample.cpu(),
                            'prediction': pred_sample.cpu()
                        })

                    metrics = calculate_all_metrics(
                        pred_sample,
                        target_sample,
                        self.sample_rate
                    )
                    metrics['sample_idx'] = batch_idx * batch_size + i
                    all_metrics.append(metrics)
                    
                    # Optionally save predictions
                    if save_predictions:
                        try:
                            # Ensure we have valid audio before saving
                            if torch.isnan(pred_sample).any():
                                print(f"Skipping save for sample {metrics['sample_idx']} due to NaNs")
                            else:
                                sample_path = output_dir / f"sample_{metrics['sample_idx']:04d}_predicted.wav"
                                self._save_audio(pred_sample.cpu(), sample_path)
                        except Exception as e:
                            print(f"Error saving audio sample {metrics['sample_idx']}: {e}")
                    
                    samples_processed += 1
        
        # Aggregate statistics
        aggregated = self._aggregate_metrics(all_metrics)
        
        # Create summary
        summary = self._create_summary(aggregated)
        
        return {
            'metrics': aggregated,
            'per_sample_metrics': all_metrics,
            'audio_samples': audio_samples,
            'summary': summary
        }
    
    def _aggregate_metrics(self, all_metrics: List[Dict]) -> Dict:
        """Compute mean, std, min, max for each metric."""
        if not all_metrics:
            return {}
        
        # Get metric names (exclude sample_idx)
        metric_names = [k for k in all_metrics[0].keys() if k != 'sample_idx']
        
        aggregated = {}
        for metric_name in metric_names:
            values = [m[metric_name] for m in all_metrics]
            aggregated[metric_name] = {
                'mean': float(torch.tensor(values).mean().item()),
                'std': float(torch.tensor(values).std().item()),
                'min': float(min(values)),
                'max': float(max(values)),
            }
        
        return aggregated
    
    def _create_summary(self, aggregated: Dict) -> str:
        """Create human-readable summary of results."""
        lines = ["=" * 60]
        lines.append("MODEL EVALUATION SUMMARY")
        lines.append("=" * 60)
        lines.append("")
        
        # Group metrics by category
        time_domain = ['mse', 'mae', 'esr', 'pre_emphasis_esr']
        freq_domain = [
            'spectral_convergence', 'multi_scale_spectral',
            'frequency_response_error_db', 'phase_response_error_rad'
        ]
        harmonic = ['impulse_response_similarity', 'thd_difference_pct']
        
        def format_metrics(title, metric_list):
            lines.append(f"{title}:")
            lines.append("-" * 60)
            for metric in metric_list:
                if metric in aggregated:
                    stats = aggregated[metric]
                    lines.append(
                        f"  {metric:30s}: "
                        f"mean={stats['mean']:8.4f} "
                        f"std={stats['std']:8.4f} "
                        f"[{stats['min']:8.4f}, {stats['max']:8.4f}]"
                    )
            lines.append("")
        
        format_metrics("TIME-DOMAIN METRICS", time_domain)
        format_metrics("FREQUENCY-DOMAIN METRICS", freq_domain)
        format_metrics("HARMONIC METRICS", harmonic)
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def save_results(
        self,
        results: Dict,
        output_path: Path,
        format: str = 'json'
    ):
        """
        Save evaluation results to file.
        
        Args:
            results: Results dictionary from evaluate()
            output_path: Path to save file
            format: 'json' or 'csv'
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        elif format == 'csv':
            # Save aggregated metrics
            aggregated_path = output_path.with_suffix('.aggregated.csv')
            with open(aggregated_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Mean', 'Std', 'Min', 'Max'])
                for metric, stats in results['metrics'].items():
                    writer.writerow([
                        metric,
                        stats['mean'],
                        stats['std'],
                        stats['min'],
                        stats['max']
                    ])
            
            # Save per-sample metrics
            per_sample_path = output_path.with_suffix('.per_sample.csv')
            if results['per_sample_metrics']:
                with open(per_sample_path, 'w', newline='') as f:
                    fieldnames = list(results['per_sample_metrics'][0].keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(results['per_sample_metrics'])
            
            print(f"Saved aggregated metrics to {aggregated_path}")
            print(f"Saved per-sample metrics to {per_sample_path}")
        
        else:
            raise ValueError(f"Unknown format: {format}. Use 'json' or 'csv'")
        
        # Always save text summary
        summary_path = output_path.with_suffix('.txt')
        with open(summary_path, 'w') as f:
            f.write(results['summary'])
        
        print(f"Results saved to {output_path}")
    
    def _save_audio(self, waveform: torch.Tensor, path: Path):
        """Save audio waveform to file."""
        import torchaudio
        torchaudio.save(str(path), waveform.cpu(), self.sample_rate)
    
    def compare_models(
        self,
        other_evaluator: 'ModelEvaluator',
        metric_name: str = 'esr'
    ) -> Dict:
        """
        Compare this model with another model.
        
        Args:
            other_evaluator: Another ModelEvaluator instance
            metric_name: Metric to compare
        
        Returns:
            Comparison dictionary with relative improvements
        """
        results_self = self.evaluate()
        results_other = other_evaluator.evaluate()
        
        self_value = results_self['metrics'][metric_name]['mean']
        other_value = results_other['metrics'][metric_name]['mean']
        
        improvement = ((other_value - self_value) / other_value) * 100
        
        return {
            'metric': metric_name,
            'model_1_value': self_value,
            'model_2_value': other_value,
            'improvement_pct': improvement,
            'better_model': 'model_1' if self_value < other_value else 'model_2'
        }
