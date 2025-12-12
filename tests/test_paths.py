"""
Tests for centralized path manager.
"""
import pytest
from pathlib import Path
from src.config.paths import ProjectPaths, paths


def test_project_root():
    """Test that project root is correctly identified."""
    assert ProjectPaths.ROOT.exists()
    assert (ProjectPaths.ROOT / "src").exists()
    assert (ProjectPaths.ROOT / "tests").exists()


def test_resource_paths():
    """Test resource directory paths."""
    assert ProjectPaths.RESOURCES == ProjectPaths.ROOT / "resources"
    assert ProjectPaths.DATASETS == ProjectPaths.RESOURCES / "datasets"
    assert ProjectPaths.CHECKPOINTS == ProjectPaths.RESOURCES / "checkpoints"


def test_output_paths():
    """Test output directory paths."""
    assert ProjectPaths.RUNS == ProjectPaths.ROOT / "runs"
    assert ProjectPaths.OUTPUTS == ProjectPaths.ROOT / "outputs"
    assert ProjectPaths.KAGGLE_OUTPUTS == ProjectPaths.OUTPUTS / "kaggle_runs"


def test_get_dataset_path():
    """Test getting dataset paths."""
    dataset_path = ProjectPaths.get_dataset_path("EGFxDataset")
    assert dataset_path == ProjectPaths.DATASETS / "EGFxDataset"
    assert str(dataset_path).endswith("EGFxDataset")


def test_get_checkpoint_path():
    """Test getting checkpoint paths."""
    # Final model path
    final_path = ProjectPaths.get_checkpoint_path("test_model")
    assert final_path == ProjectPaths.CHECKPOINTS / "test_model" / "final_model.pt"
    
    # Epoch checkpoint path
    epoch_path = ProjectPaths.get_checkpoint_path("test_model", epoch=42)
    assert epoch_path == ProjectPaths.CHECKPOINTS / "test_model" / "checkpoint_epoch_42.pt"


def test_get_run_path():
    """Test getting run paths."""
    run_path = ProjectPaths.get_run_path("my_experiment")
    assert run_path == ProjectPaths.RUNS / "my_experiment"


def test_to_str():
    """Test Path to string conversion."""
    path = ProjectPaths.RUNS / "test"
    path_str = ProjectPaths.to_str(path)
    assert isinstance(path_str, str)
    assert "runs" in path_str
    assert "test" in path_str


def test_get_relative_to_root():
    """Test getting relative paths."""
    abs_path = ProjectPaths.RUNS / "experiment"
    rel_path = ProjectPaths.get_relative_to_root(abs_path)
    
    # Should be relative to root
    assert not rel_path.startswith(str(ProjectPaths.ROOT))
    assert "runs" in rel_path
    assert "experiment" in rel_path


def test_singleton_instance():
    """Test that paths singleton works."""
    assert paths.ROOT == ProjectPaths.ROOT
    assert paths.DATASETS == ProjectPaths.DATASETS


def test_ensure_directories(tmp_path, monkeypatch):
    """Test directory creation."""
    # Temporarily change ROOT to tmp_path for testing
    monkeypatch.setattr(ProjectPaths, 'ROOT', tmp_path)
    monkeypatch.setattr(ProjectPaths, 'RESOURCES', tmp_path / "resources")
    monkeypatch.setattr(ProjectPaths, 'DATASETS', tmp_path / "resources" / "datasets")
    monkeypatch.setattr(ProjectPaths, 'CHECKPOINTS', tmp_path / "resources" / "checkpoints")
    monkeypatch.setattr(ProjectPaths, 'RUNS', tmp_path / "runs")
    monkeypatch.setattr(ProjectPaths, 'OUTPUTS', tmp_path / "outputs")
    monkeypatch.setattr(ProjectPaths, 'KAGGLE_OUTPUTS', tmp_path / "outputs" / "kaggle_runs")
    monkeypatch.setattr(ProjectPaths, 'KAGGLE_BUILD', tmp_path / "kaggle_build")
    
    ProjectPaths.ensure_directories()
    
    # Verify directories were created
    assert (tmp_path / "resources").exists()
    assert (tmp_path / "resources" / "datasets").exists()
    assert (tmp_path / "resources" / "checkpoints").exists()
    assert (tmp_path / "runs").exists()
    assert (tmp_path / "outputs").exists()
    assert (tmp_path / "outputs" / "kaggle_runs").exists()
