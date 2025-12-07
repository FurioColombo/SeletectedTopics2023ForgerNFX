from pydantic import BaseModel, Field
from typing import Optional, List

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
    
class ProjectConfig(BaseModel):
    audio: AudioConfig = Field(default_factory=AudioConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    
    data_dir: str = "data"
    output_dir: str = "runs"

def load_config(path: Optional[str] = None) -> ProjectConfig:
    # TODO: Implement loading from YAML
    return ProjectConfig()
