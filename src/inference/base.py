from abc import ABC, abstractmethod
from typing import Generator, Tuple, Any, Dict
import torch
from ..models.base import BaseAudioModel

class InferenceRunner(ABC):
    """
    Abstract base class for running inference on audio data.
    """
    
    def __init__(self, model: BaseAudioModel, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    @abstractmethod
    def run(self) -> Generator[Tuple, None, None]:
        """
        Generator that yields inference results.
        Format depends on the implementation.
        """
        pass
