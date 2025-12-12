import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class AudioLoss(nn.Module, ABC):
    @abstractmethod
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pass

class MSELoss(AudioLoss):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred, target)

class ESRLoss(AudioLoss):
    """Error Signal Ratio Loss"""
    def __init__(self):
        super().__init__()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (batch, channels, time)
        error = target - pred
        numerator = torch.sum(error ** 2, dim=2)
        denominator = torch.sum(target ** 2, dim=2) + 1e-8
        return torch.mean(numerator / denominator)

class CombinedLoss(AudioLoss):
    def __init__(self, losses: dict[AudioLoss, float]):
        super().__init__()
        self.losses = losses
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for loss_fn, weight in self.losses.items():
            total_loss += weight * loss_fn(pred, target)
        return total_loss

class MultiScaleSpectralLoss(AudioLoss):
    """
    Multi-Scale Spectral Loss.
    Computes L1 distance between magnitude spectrograms at multiple resolutions.
    """
    def __init__(self, fft_sizes=(2048, 1024, 512, 256, 128, 64)):
        super().__init__()
        self.fft_sizes = fft_sizes

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        
        # Ensure we are working with correct shapes, sometimes (batch, channels, time) needs reshaping
        # metrics.py logic: pred.reshape(-1, pred.shape[-1]).squeeze()
        # simplified here assuming standard (B, C, T) or (B, T)
        
        p = pred.reshape(-1, pred.shape[-1])
        t = target.reshape(-1, target.shape[-1])
        
        # Get input length to validate FFT sizes
        input_length = p.shape[-1]
        
        # Filter FFT sizes to only use those that won't cause padding errors
        # torch.stft pads by n_fft//2 on each side, so we need input_length >= n_fft//2
        valid_fft_sizes = [n_fft for n_fft in self.fft_sizes if n_fft <= input_length]
        
        if not valid_fft_sizes:
            # Fallback: use the largest power of 2 smaller than or equal to input length
            import math
            if input_length >= 64:
                max_fft = 2 ** int(math.log2(input_length))
                valid_fft_sizes = [max_fft]
            else:
                # For very small inputs, use the input length itself (rounded down to power of 2)
                max_fft = 2 ** int(math.log2(max(input_length, 2)))  # minimum FFT size of 2
                valid_fft_sizes = [max_fft]

        for n_fft in valid_fft_sizes:
            hop_length = n_fft // 4
            window = torch.hann_window(n_fft, device=pred.device)
            
            p_stft = torch.stft(
                p,
                n_fft=n_fft,
                hop_length=hop_length,
                window=window,
                return_complex=True
            )
            t_stft = torch.stft(
                t,
                n_fft=n_fft,
                hop_length=hop_length,
                window=window,
                return_complex=True
            )
            
            p_mag = torch.abs(p_stft)
            t_mag = torch.abs(t_stft)
            
            # L1 Loss on Magnitude
            loss = torch.mean(torch.abs(p_mag - t_mag))
            
            # Optionally Log-Magnitude loss can be added here
            # p_log = torch.log(p_mag + 1e-7)
            # t_log = torch.log(t_mag + 1e-7)
            # loss += torch.mean(torch.abs(p_log - t_log)) 
            
            total_loss += loss
            
        return total_loss / len(valid_fft_sizes)
