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
        # Flatten structure: test/quantitative/esr_mean
        for metric_name, stats in results['metrics'].items():
            if isinstance(stats, dict):
                for stat_name, val in stats.items():
                    # prioritizing mean for main view
                    if stat_name == 'mean':
                        log_dict[f"test/quantitative/{metric_name}"] = val
                    log_dict[f"test/quantitative/{metric_name}_{stat_name}"] = val
            else:
                 log_dict[f"test/quantitative/{metric_name}"] = stats
                 
        wandb.log(log_dict)
        
    def log_test_qualitative(self, sequences: List[Dict[str, Any]], visualizer=None):
        if not sequences:
            return
            
        print("📤 Uploading qualitative results to W&B...")
        
        # Create Table
        columns = ["name", "audio_input", "audio_target", "audio_prediction", "spectrogram_overlap"]
        table = wandb.Table(columns=columns)
        
        for seq in sequences:
            name = seq['name']
            sr = seq['sample_rate']
            
            # Convert tensors to numpy if needed
            inp = seq['input'].numpy() if torch.is_tensor(seq['input']) else seq['input']
            tgt = seq['target'].numpy() if torch.is_tensor(seq['target']) else seq['target']
            pred = seq['prediction'].numpy() if torch.is_tensor(seq['prediction']) else seq['prediction']
            
            # Create Plot if visualizer provided
            plot_html = None
            if visualizer:
                # Convert back to tensor for visualizer if needed, or update visualizer to handle numpy
                # Visualizer expects tensors currently
                inp_t = torch.tensor(inp) if not torch.is_tensor(seq['input']) else seq['input']
                tgt_t = torch.tensor(tgt) if not torch.is_tensor(seq['target']) else seq['target']
                pred_t = torch.tensor(pred) if not torch.is_tensor(seq['prediction']) else seq['prediction']
                
                fig = visualizer.plot_spectral_overlap(inp_t, pred_t, tgt_t)
                plot_html = wandb.Html(fig.to_html(include_plotlyjs='cdn'))
            
            table.add_data(
                name,
                wandb.Audio(inp, sample_rate=sr, caption="Input"),
                wandb.Audio(tgt, sample_rate=sr, caption="Target"),
                wandb.Audio(pred, sample_rate=sr, caption="Prediction"),
                plot_html
            )
            
        wandb.log({"test/qualitative_analysis": table})
        
    def finish(self):
        wandb.finish()
