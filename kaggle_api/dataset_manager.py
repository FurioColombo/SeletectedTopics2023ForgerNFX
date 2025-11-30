import os
import json
from pathlib import Path
from typing import Optional
from .auth import get_api_client
from .paths import LOCAL_DATA_ROOT

class DatasetManager:
    def __init__(self, config_path: str = "config/dataset.yaml"):
        self.api = get_api_client()
        self.config_path = Path(config_path)
        
    def _load_config(self):
        import yaml
        if not self.config_path.exists():
            raise FileNotFoundError(f"Dataset config not found at {self.config_path}")
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def validate_dataset(self):
        """Checks if local dataset matches expected structure."""
        config = self._load_config()
        local_root = Path(config.get('local_root', LOCAL_DATA_ROOT))
        
        if not local_root.exists():
            raise FileNotFoundError(f"Local dataset root {local_root} does not exist.")
            
        # Check expected subfolders
        expected_folders = config.get('expected_folders', [])
        for folder in expected_folders:
            if not (local_root / folder).exists():
                raise FileNotFoundError(f"Expected subfolder {folder} missing in {local_root}")
                
        print("Dataset validation passed.")
        return config

    def push_dataset(self):
        """Uploads or updates the dataset on Kaggle."""
        config = self.validate_dataset()
        slug = config['slug']
        local_root = Path(config.get('local_root', LOCAL_DATA_ROOT))
        
        # Check if dataset exists
        try:
            self.api.dataset_status(slug)
            exists = True
        except Exception:
            exists = False
            
        # Prepare metadata
        meta_file = local_root / "dataset-metadata.json"
        if not meta_file.exists():
            metadata = {
                "title": config['title'],
                "id": slug,
                "licenses": [{"name": "CC0-1.0"}]
            }
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        if exists:
            print(f"Updating dataset {slug}...")
            self.api.dataset_create_version(
                folder=str(local_root),
                version_notes="Automated update via CLI",
                dir_mode="zip"
            )
        else:
            print(f"Creating new dataset {slug}...")
            self.api.dataset_create_new(
                folder=str(local_root),
                dir_mode="zip",
                public=False
            )
        print("Dataset push initiated.")
