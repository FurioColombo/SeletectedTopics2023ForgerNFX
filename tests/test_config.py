import pytest
from src.config.config import load_config, ProjectConfig

def test_default_config():
    config = load_config()
    assert isinstance(config, ProjectConfig)
    assert config.audio.sample_rate == 44100
    assert config.model.name == "lstm"
    assert config.training.batch_size == 32

def test_config_structure():
    config = ProjectConfig()
    assert hasattr(config, 'audio')
    assert hasattr(config, 'model')
    assert hasattr(config, 'training')
