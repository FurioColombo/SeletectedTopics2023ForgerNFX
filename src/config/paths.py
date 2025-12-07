"""
Centralized path management for the project.

This module provides a single source of truth for all project paths,
making it easy to reorganize directories without breaking code.
"""
from pathlib import Path
from typing import Optional


class ProjectPaths:
    """Centralized path configuration for the entire project."""
    
    # Project root directory (3 levels up from this file: src/config/paths.py)
    ROOT = Path(__file__).resolve().parent.parent.parent
    
    # Main directories
    RESOURCES = ROOT / "resources"
    DATASETS = RESOURCES / "datasets"
    CHECKPOINTS = RESOURCES / "checkpoints"
    
    # Output directories
    RUNS = ROOT / "runs"
    OUTPUTS = ROOT / "outputs"
    KAGGLE_OUTPUTS = OUTPUTS / "kaggle_runs"
    KAGGLE_BUILD = ROOT / "kaggle_build"
    
    # Config directory
    CONFIG = ROOT / "config"
    
    # Source code
    SRC = ROOT / "src"
    
    # Tests
    TESTS = ROOT / "tests"
    
    # Training directory (legacy)
    TRAINING = ROOT / "training"
    
    @classmethod
    def get_dataset_path(cls, dataset_name: str) -> Path:
        """Get path to a specific dataset."""
        return cls.DATASETS / dataset_name
    
    @classmethod
    def get_checkpoint_path(cls, model_name: str, epoch: Optional[int] = None) -> Path:
        """Get path to model checkpoints."""
        checkpoint_dir = cls.CHECKPOINTS / model_name
        if epoch is not None:
            return checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        return checkpoint_dir / "final_model.pt"
    
    @classmethod
    def get_run_path(cls, run_name: str) -> Path:
        """Get path to a specific training run."""
        return cls.RUNS / run_name
    
    @classmethod
    def ensure_directories(cls):
        """Create all necessary directories if they don't exist."""
        directories = [
            cls.RESOURCES,
            cls.DATASETS,
            cls.CHECKPOINTS,
            cls.RUNS,
            cls.OUTPUTS,
            cls.KAGGLE_OUTPUTS,
            cls.KAGGLE_BUILD,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def to_str(cls, path: Path) -> str:
        """Convert Path to string (for backward compatibility)."""
        return str(path)
    
    @classmethod
    def get_relative_to_root(cls, path: Path) -> str:
        """Get path relative to project root as string."""
        try:
            return str(path.relative_to(cls.ROOT))
        except ValueError:
            # Path is not relative to ROOT
            return str(path)


# For convenience, create a singleton instance
paths = ProjectPaths()
