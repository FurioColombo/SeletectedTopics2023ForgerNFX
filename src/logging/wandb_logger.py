from typing import Dict, Any, List, Optional
import wandb
import torch
import numpy as np
from pathlib import Path
from .base import BaseLogger

class WandbLogger(BaseLogger):
    """
    Logger implementation for Weights & Biases.
    Organizes logs into sections: train/, val/, test/quantitative/, test/qualitative/.
    """
    
    def __init__(self, project: str, config: Dict[str, Any], name: Optional[str] = None, tags: Optional[List[str]] = None):
        """
        Initialize W&B run.
        """
        self.run = wandb.init(
            project=project,
            config=config,
            name=name,
            tags=tags,
            reinit=True
        )
        
    def log_training(self, epoch: int, batch: int, loss: float, lr: float):
        wandb.log({
            "train/loss": loss,
            "train/lr": lr,
            "train/epoch": epoch,
            "train/batch": batch
        })
        
    def log_validation(self, epoch: int, avg_loss: float, metrics: Dict[str, float]):
        log_dict = {
            "val/loss": avg_loss,
            "val/epoch": epoch
        }
        # Add other metrics with val/ prefix
        for k, v in metrics.items():
            log_dict[f"val/{k}"] = v
            
        wandb.log(log_dict)
        
    def log_test_quantitative(self, results: Dict[str, Any]):
        """
        Expects results dict from MetricAnalyzer.get_aggregated_results()
        Structure: {'metrics': {'esr': {'mean': 0.1, ...}, ...}, ...}
        """
        if 'metrics' not in results:
            return
            
        log_dict = {}
        
        # Key metrics to track (keep it clean)
        key_metrics = [
            'esr', 
            'mse', 
            'phase_response_error_rad', 
            'multi_scale_spectral', 
            'thd_difference_pct',
            'frequency_response_error_db'
        ]
        
        # Flatten structure: test/quantitative/esr
        for metric_name, stats in results['metrics'].items():
            # Only log key metrics
            if metric_name not in key_metrics:
                continue
                
            if isinstance(stats, dict):
                # Log only mean for clean dashboard
                if 'mean' in stats:
                    log_dict[f"test/quantitative/{metric_name}"] = stats['mean']
            else:
                 log_dict[f"test/quantitative/{metric_name}"] = stats
                 
        wandb.log(log_dict)
        
    def _prepare_audio_for_wandb(self, audio_data):
        """Helper to convert audio to format accepted by wandb.Audio (numpy, (time,), float32)"""
        # 1. Convert to numpy
        if torch.is_tensor(audio_data):
            audio_data = audio_data.detach().cpu().float().numpy()
        
        # 2. Squeeze extra dimensions (1, T) -> (T,)
        if audio_data.ndim > 1:
            # If shape is (Channels, Time) and Channels is 1, squeeze it
            if audio_data.shape[0] == 1:
                audio_data = audio_data.flatten()
            elif audio_data.shape[0] < audio_data.shape[1]: 
                # Likely (C, T) -> Transpose to (T, C) for soundfile/wandb if stereo
                audio_data = audio_data.T
                
        return audio_data

    def log_test_qualitative(self, sequences: List[Dict[str, Any]], visualizer=None):
        if not sequences:
            return
            
        print("📤 Uploading qualitative results to W&B...")
        
        # Create Table
        columns = ["name", "audio_input", "audio_target", "audio_prediction", "spectrogram_overlap"]
        table = wandb.Table(columns=columns)
        
        # Also log plots as standalone images for better visibility
        plot_dict = {}
        
        for idx, seq in enumerate(sequences):
            name = seq['name']
            sr = seq['sample_rate']
            
            # Prepare audio for W&B
            inp_np = self._prepare_audio_for_wandb(seq['input'])
            tgt_np = self._prepare_audio_for_wandb(seq['target'])
            pred_np = self._prepare_audio_for_wandb(seq['prediction'])
            
            # Create Plot if visualizer provided
            plot_html = None
            if visualizer:
                # Visualizer expects tensors (likely (1, T) or (C, T))
                # We reuse the raw seq data which we know visualizer handles (or handled before)
                # But to be safe, let's pass tensor versions of what we just cleaned clean, 
                # ensuring visualizer gets standard input.
                # Actually, visualizer.plot_spectral_overlap expects Tensor (C, T) or (T).
                # Let's trust visualizer handles the original seq['input'] tensors fine as it did in the traceback before hitting wandb.Audio
                
                inp_t = seq['input'] if torch.is_tensor(seq['input']) else torch.from_numpy(seq['input'])
                tgt_t = seq['target'] if torch.is_tensor(seq['target']) else torch.from_numpy(seq['target'])
                pred_t = seq['prediction'] if torch.is_tensor(seq['prediction']) else torch.from_numpy(seq['prediction'])

                # Generate spectral overlap plot
                fig = visualizer.plot_spectral_overlap(inp_t, pred_t, tgt_t)
                plot_html = wandb.Html(fig.to_html(include_plotlyjs='cdn'))
                
                # Also log as standalone plotly chart (more visible in W&B)
                plot_dict[f"test/qualitative/spectral_overlap/{name}"] = fig
            
            # Log individual audio files as well (easier to find than in table)
            plot_dict[f"test/qualitative/audio/{name}/input"] = wandb.Audio(inp_np, sample_rate=sr, caption=f"{name}_input")
            plot_dict[f"test/qualitative/audio/{name}/target"] = wandb.Audio(tgt_np, sample_rate=sr, caption=f"{name}_target")
            plot_dict[f"test/qualitative/audio/{name}/prediction"] = wandb.Audio(pred_np, sample_rate=sr, caption=f"{name}_prediction")
            
            table.add_data(
                name,
                wandb.Audio(inp_np, sample_rate=sr, caption="Input"),
                wandb.Audio(tgt_np, sample_rate=sr, caption="Target"),
                wandb.Audio(pred_np, sample_rate=sr, caption="Prediction"),
                plot_html
            )
        
        # Log everything at once
        plot_dict["test/qualitative_analysis"] = table
        wandb.log(plot_dict)
        
    def finish(self):
        wandb.finish()
