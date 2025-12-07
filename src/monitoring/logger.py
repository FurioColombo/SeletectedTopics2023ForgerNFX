"""
Centralized monitoring and logging for model training.

Provides a unified interface for experiment tracking with fallback support.
Supports Weights & Biases (wandb) with graceful degradation to local logging
if wandb is unavailable or disabled.
"""
import os
from typing import Dict, Optional, Any
from pathlib import Path
import json
import torch


class MonitoringLogger:
    """
    Centralized experiment tracking and monitoring.
    
    Automatically uses wandb if available, falls back to local logging otherwise.
    """
    
    def __init__(
        self,
        project: str = "forger-nfx",
        run_name: Optional[str] = None,
        config: Optional[Dict] = None,
        use_wandb: bool = True,
        log_dir: Optional[Path] = None
    ):
        """
        Initialize monitoring logger.
        
        Args:
            project: Project name for wandb
            run_name: Name for this specific run
            config: Configuration dictionary to log
            use_wandb: Whether to attempt using wandb
            log_dir: Directory for local logs (fallback)
        """
        self.use_wandb = use_wandb
        self.wandb_available = False
        self.run = None
        
        # Try to initialize wandb
        if use_wandb:
            try:
                import wandb
                
                # Check if wandb API key is set (for Kaggle)
                if 'WANDB_API_KEY' in os.environ or os.path.exists(Path.home() / '.netrc'):
                    self.run = wandb.init(
                        project=project,
                        name=run_name,
                        config=config or {},
                        reinit=True
                    )
                    self.wandb_available = True
                    print(f"✅ Wandb initialized: {wandb.run.url}")
                else:
                    # Try anonymous mode for quick start
                    self.run = wandb.init(
                        project=project,
                        name=run_name,
                        config=config or {},
                        anonymous="allow",
                        reinit=True
                    )
                    self.wandb_available = True
                    print(f"✅ Wandb initialized (anonymous mode): {wandb.run.url}")
                    
            except Exception as e:
                print(f"⚠️  Wandb not available: {e}")
                print("   Falling back to local logging")
                self.wandb_available = False
        
        # Setup local logging fallback
        if log_dir is None:
            from ..config.paths import paths
            log_dir = paths.OUTPUTS / "local_logs" / (run_name or "default")
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_log = self.log_dir / "metrics.jsonl"
        self.config_file = self.log_dir / "config.json"
        
        # Save config locally
        if config:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        
        print(f"📁 Local logs: {self.log_dir}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step/epoch number
        """
        # Log to wandb
        if self.wandb_available and self.run:
            self.run.log(metrics, step=step)
        
        # Log locally (always)
        log_entry = metrics.copy()
        if step is not None:
            log_entry['step'] = step
        
        with open(self.metrics_log, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_audio(
        self,
        audio_path: str,
        caption: str = "",
        sample_rate: int = 44100
    ):
        """
        Log audio file.
        
        Args:
            audio_path: Path to audio file
            caption: Description of audio
            sample_rate: Sample rate
        """
        if self.wandb_available and self.run:
            try:
                import wandb
                self.run.log({
                    caption or "audio": wandb.Audio(
                        audio_path,
                        sample_rate=sample_rate,
                        caption=caption
                    )
                })
            except Exception as e:
                print(f"⚠️  Could not log audio to wandb: {e}")
        
        # Copy audio to local logs
        import shutil
        local_audio = self.log_dir / Path(audio_path).name
        shutil.copy(audio_path, local_audio)
    
    def log_model(self, model_path: str, name: str = "model"):
        """
        Log model checkpoint.
        
        Args:
            model_path: Path to model file
            name: Name for the model artifact
        """
        if self.wandb_available and self.run:
            try:
                import wandb
                artifact = wandb.Artifact(name, type='model')
                artifact.add_file(model_path)
                self.run.log_artifact(artifact)
            except Exception as e:
                print(f"⚠️  Could not log model to wandb: {e}")
        
        print(f"💾 Model saved locally: {model_path}")
    
    def log_image(self, image_path: str, caption: str = ""):
        """
        Log image (e.g., plots, spectrograms).
        
        Args:
            image_path: Path to image file
            caption: Description
        """
        if self.wandb_available and self.run:
            try:
                import wandb
                self.run.log({caption or "image": wandb.Image(image_path, caption=caption)})
            except Exception as e:
                print(f"⚠️  Could not log image to wandb: {e}")
        
        # Copy image to local logs
        import shutil
        local_image = self.log_dir / Path(image_path).name
        shutil.copy(image_path, local_image)
    
    def watch_model(self, model: torch.nn.Module):
        """
        Watch model for gradient and parameter tracking.
        
        Args:
            model: PyTorch model to watch
        """
        if self.wandb_available and self.run:
            try:
                import wandb
                wandb.watch(model, log='all', log_freq=100)
            except Exception as e:
                print(f"⚠️  Could not watch model: {e}")
    
    def finish(self):
        """Clean up and finish logging."""
        if self.wandb_available and self.run:
            try:
                self.run.finish()
            except:
                pass
        
        print(f"✅ Logging complete. Local logs at: {self.log_dir}")
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.finish()


def create_kaggle_logger(
    project: str = "forger-nfx",
    run_name: Optional[str] = None,
    config: Optional[Dict] = None
) -> MonitoringLogger:
    """
    Create a logger configured for Kaggle environment.
    
    Automatically reads WANDB_API_KEY from Kaggle secrets if available.
    
    Args:
        project: Project name
        run_name: Run name
        config: Configuration dict
    
    Returns:
        Configured MonitoringLogger
    """
    # On Kaggle, secrets are accessed via UserSecretsClient
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        api_key = user_secrets.get_secret("wandb_api_key")
        if api_key:
            os.environ['WANDB_API_KEY'] = api_key
            print("✅ Loaded wandb API key from Kaggle secrets")
    except ImportError:
        # Not on Kaggle or library missing
        pass
    except Exception as e:
        print(f"⚠️  Could not load wandb secret: {e}")
    
    return MonitoringLogger(
        project=project,
        run_name=run_name,
        config=config,
        use_wandb=True
    )
