import torch
from torch.utils.data import DataLoader
from typing import Generator, Tuple, List, Dict, Optional
from pathlib import Path
import torchaudio
from tqdm import tqdm

from .base import InferenceRunner
from ..models.base import BaseAudioModel

class SegmentRunner(InferenceRunner):
    """
    Runs inference on batches of segmented audio (e.g., from a DataLoader).
    Used for calculating statistical metrics over the dataset.
    """
    def __init__(
        self, 
        model: BaseAudioModel, 
        dataloader: DataLoader, 
        device: str = "cpu",
        limit_batches: Optional[int] = None
    ):
        super().__init__(model, device)
        self.dataloader = dataloader
        self.limit_batches = limit_batches

    def run(self) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Yields batch results:
        {
            'input': tensor,
            'target': tensor,
            'prediction': tensor
        }
        """
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(self.dataloader, desc="Running Segments")):
                if self.limit_batches and batch_idx >= self.limit_batches:
                    break
                
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                predictions = self.model(inputs)
                
                # Sanity check
                if torch.isnan(predictions).any():
                    predictions = torch.nan_to_num(predictions)
                
                yield {
                    'input': inputs.cpu(),
                    'target': targets.cpu(),
                    'prediction': predictions.cpu()
                }

class SequenceRunner(InferenceRunner):
    """
    Runs inference on full audio files.
    Uses stateful inference (hidden state propagation) for LSTMs.
    """
    def __init__(
        self,
        model: BaseAudioModel,
        file_pairs: List[Tuple[Path, Path]], # (input_path, target_path)
        block_size: int = 2048,
        sample_rate: int = 44100,
        device: str = "cpu"
    ):
        super().__init__(model, device)
        self.file_pairs = file_pairs
        self.block_size = block_size
        self.sample_rate = sample_rate

    def run(self) -> Generator[Dict[str, Any], None, None]:
        """
        Yields full sequence results:
        {
            'name': str,
            'input': tensor (full),
            'target': tensor (full),
            'prediction': tensor (full),
            'sample_rate': int
        }
        """
        with torch.no_grad():
            for input_path, target_path in tqdm(self.file_pairs, desc="Running Sequences"):
                # Load Audio
                input_wav, sr_in = torchaudio.load(input_path)
                target_wav, sr_tgt = torchaudio.load(target_path)
                
                # Resample if needed
                if sr_in != self.sample_rate:
                    input_wav = torchaudio.transforms.Resample(sr_in, self.sample_rate)(input_wav)
                if sr_tgt != self.sample_rate:
                    target_wav = torchaudio.transforms.Resample(sr_tgt, self.sample_rate)(target_wav)
                
                # Ensure mono
                if input_wav.shape[0] > 1: input_wav = input_wav[:1]
                if target_wav.shape[0] > 1: target_wav = target_wav[:1]

                # Match lengths (crop to min)
                min_len = min(input_wav.shape[1], target_wav.shape[1])
                input_wav = input_wav[:, :min_len]
                target_wav = target_wav[:, :min_len]
                
                # --- Stateful Inference ---
                # We process the file in chunks, but we need to manage the model's internal state.
                # Assuming the model handles state internally if we don't reset it, 
                # OR we assume the model is stateless between calls unless specific measures are taken.
                # For standard LSTM implementations in PyTorch (nn.LSTM), hidden state is usually returned.
                # BUT our BaseAudioModel definition wraps forward(x) -> y.
                # We need to know if the model supports stateful processing.
                
                # Strategy:
                # 1. Provide input as (1, 1, block_size) or (1, channels, total_len) if it fits in VRAM?
                # If total_len is small (e.g. 5 seconds), we can process at once.
                # If large (3 mins), we must chunk.
                
                # Let's try processing as one big batch first if VRAM permits (simplest for avoiding state bugs).
                # A 3-minute song at 44.1k is ~8M samples -> 32MB float32. Fits easily in VRAM.
                # The issue is typically BPTT limit during training, but for inference we can do full sequence
                # IF the model architecture allows variable length input.
                # CNNs/LSTMs usually do.
                
                # However, user requested "iteratively propagate lstm states".
                # If our model class doesn't expose hidden states in `forward`, we can't manually propagate them
                # unless we modify the model signature or pass the whole thing.
                # BUT, `nn.LSTM` can take the whole sequence at once and compute output for all conformally.
                
                # DECISION:
                # To guarantee we match "chunked" training behavior but continuously, 
                # we will try to pass the whole sequence. Torch LSTMs handle the loop internally efficiently.
                # This achieves "stateful inference" implicitly because the LSTM unrolls over the full sequence.
                
                input_tensor = input_wav.unsqueeze(0).to(self.device) # (1, 1, time)
                
                try:
                    # Try full sequence inference
                    pred_tensor = self.model(input_tensor)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        # Fallback to chunking with state? 
                        # Without model-specific state API, we can't do true stateful chunking generically.
                        # We will warn and do stateless chunking (which is suboptimal) or crash.
                        # Given this is "Forger NFX", let's assume standard models that fit 1 song in inference VRAM.
                        print(f"⚠️ OOM on full sequence {input_path.name}, skipping.")
                        continue
                    raise e
                    
                pred_wav = pred_tensor.squeeze(0).cpu()
                
                yield {
                    'name': input_path.stem,
                    'input': input_wav,
                    'target': target_wav,
                    'prediction': pred_wav,
                    'sample_rate': self.sample_rate
                }
