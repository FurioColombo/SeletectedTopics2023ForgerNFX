from torch.utils.data import DataLoader, random_split
from typing import Tuple, List
from .dataset import GuitarDataset
from project.config.config import ProjectConfig

def create_dataloaders(dataset: GuitarDataset, config: ProjectConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    total_size = len(dataset)
    val_size = int(total_size * config.training.val_split)
    test_size = int(total_size * config.training.test_split)
    train_size = total_size - val_size - test_size
    
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.training.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.training.batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
