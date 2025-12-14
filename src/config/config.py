from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import yaml
from pathlib import Path

class AudioConfig(BaseModel):
    sample_rate: int = 44100
    block_size: int = 512
    
class ModelConfig(BaseModel):
    name: str = "lstm"
    hidden_size: int = 16
    kernel_size: Optional[int] = None
    
class TrainingConfig(BaseModel):
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Scheduler
    scheduler_type: str = "warmup_plateau"
    scheduler_params: dict = Field(default_factory=lambda: {
        "warmup_steps": 100,
        "patience": 5,
        "factor": 0.5,
        "min_lr": 1e-6
    })
    
    # Early Stopping
    early_stopping: bool = False
    early_stopping_patience: int = 10
    
class ProjectConfig(BaseModel):
    audio: AudioConfig = Field(default_factory=AudioConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    
    data_dir: str = "data"
    output_dir: str = "runs"

def load_config(path: Optional[str] = None) -> ProjectConfig:
    config = ProjectConfig()
    if path and Path(path).exists():
        try:
            with open(path, "r") as f:
                yaml_data = yaml.safe_load(f)
            
            if yaml_data:
                if "audio" in yaml_data:
                    config.audio = AudioConfig(**{**config.audio.dict(), **yaml_data["audio"]})
                if "model" in yaml_data:
                    config.model = ModelConfig(**{**config.model.dict(), **yaml_data["model"]})
                if "training" in yaml_data:
                    config.training = TrainingConfig(**{**config.training.dict(), **yaml_data["training"]})
                    
                if "data_dir" in yaml_data: config.data_dir = yaml_data["data_dir"]
                if "output_dir" in yaml_data: config.output_dir = yaml_data["output_dir"]
                
                # Support flat structure overrides
                if "batch_size" in yaml_data: config.training.batch_size = yaml_data["batch_size"]
                if "epochs" in yaml_data: config.training.epochs = yaml_data["epochs"]
                if "learning_rate" in yaml_data: config.training.learning_rate = yaml_data["learning_rate"]
        except Exception as e:
            print(f"Warning: Error loading config from {path}: {e}")
            
    return config
