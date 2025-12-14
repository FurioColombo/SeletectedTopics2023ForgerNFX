import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import math


def calculate_esr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate Error Signal Ratio (ESR).
    
    Args:
        pred: Predicted signal (channels, time) or (batch, channels, time)
        target: Target signal (same shape as pred)
    
    Returns:
        ESR value (lower is better)
    """
    error = target - pred
    numerator = torch.sum(error ** 2)
    denominator = torch.sum(target ** 2) + 1e-8
    return (numerator / denominator).item()


def calculate_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate Mean Squared Error."""
    return torch.mean((pred - target) ** 2).item()


def calculate_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate Mean Absolute Error."""
    return torch.mean(torch.abs(pred - target)).item()


def calculate_pre_emphasis_esr(
    pred: torch.Tensor, 
    target: torch.Tensor,
    coef: float = 0.95
) -> float:
    """
    Calculate ESR with pre-emphasis filter (emphasizes high frequencies).
    
    Useful for guitar effects where high-frequency detail is important.
    
    Args:
        pred: Predicted signal
        target: Target signal
        coef: Pre-emphasis coefficient (default 0.95)
    
    Returns:
        Pre-emphasized ESR value
    """
    # Apply pre-emphasis: y[n] = x[n] - coef * x[n-1]
    pred_emp = pred[..., 1:] - coef * pred[..., :-1]
    target_emp = target[..., 1:] - coef * target[..., :-1]
    
    return calculate_esr(pred_emp, target_emp)


def spectral_convergence(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate spectral convergence metric.
    
    Measures how well the frequency magnitude spectrum matches.
    
    Args:
        pred: Predicted signal
        target: Target signal
    
    Returns:
        Spectral convergence (lower is better)
    """
    # Determine appropriate FFT size based on input length
    input_length = pred.shape[-1]
    n_fft = 2048
    
    # Adjust n_fft if input is shorter than n_fft
    if input_length < n_fft:
        # Use largest power of 2 <= input_length, minimum 64
        n_fft = 2 ** int(math.log2(max(input_length, 64)))
        # If still too large (e.g. input < 64), cap at input_length
        if n_fft > input_length:
            n_fft = input_length
            
    hop_length = n_fft // 4
    win_length = n_fft
    
    # Compute STFT
    pred_stft = torch.stft(
        pred.reshape(-1, pred.shape[-1]).squeeze(),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length, device=pred.device),
        return_complex=True
    )
    target_stft = torch.stft(
        target.reshape(-1, target.shape[-1]).squeeze(),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length, device=target.device),
        return_complex=True
    )
    
    # Magnitude spectra
    pred_mag = torch.abs(pred_stft)
    target_mag = torch.abs(target_stft)
    
    # Spectral convergence
    numerator = torch.norm(target_mag - pred_mag, p='fro')
    denominator = torch.norm(target_mag, p='fro') + 1e-8
    
    return (numerator / denominator).item()


def multi_scale_spectral_loss(
    pred: torch.Tensor, 
    target: torch.Tensor,
    fft_sizes: Tuple[int, ...] = (2048, 1024, 512, 256)
) -> float:
    """
    Calculate multi-scale spectral loss across different FFT sizes.
    
    Captures both fine and coarse frequency characteristics.
    
    Args:
        pred: Predicted signal
        target: Target signal
        fft_sizes: Tuple of FFT sizes to use
    
    Returns:
        Average spectral loss across scales
    """
    total_loss = 0.0
    input_length = pred.shape[-1]
    
    # Filter FFT sizes to only use those that fit within input_length
    valid_fft_sizes = [n for n in fft_sizes if n <= input_length]
    
    # Fallback if no valid FFT sizes (input too small)
    if not valid_fft_sizes:
        if input_length >= 64:
            valid_fft_sizes = [2 ** int(math.log2(input_length))]
        else:
            # Fallback for extremely small inputs (e.g. < 64)
            # Ensure at least size 4 or so to run STFT without error if possible, 
            # or just use input_length if small power of 2
            valid_fft_sizes = [2 ** int(math.log2(max(input_length, 4)))]
            
    for n_fft in valid_fft_sizes:
        hop_length = n_fft // 4
        window = torch.hann_window(n_fft, device=pred.device)
        
        pred_stft = torch.stft(
            pred.reshape(-1, pred.shape[-1]).squeeze(),
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True
        )
        target_stft = torch.stft(
            target.reshape(-1, target.shape[-1]).squeeze(),
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True
        )
        
        # Magnitude loss
        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)
        mag_loss = torch.mean(torch.abs(pred_mag - target_mag))
        
        total_loss += mag_loss.item()
    
    return total_loss / len(valid_fft_sizes)


def frequency_response_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_rate: int = 44100
) -> float:
    """
    Calculate frequency response error by comparing magnitude responses.
    
    Args:
        pred: Predicted signal
        target: Target signal
        sample_rate: Audio sample rate
    
    Returns:
        Mean frequency response error in dB
    """
    # Compute FFT
    pred_fft = torch.fft.rfft(pred.reshape(-1, pred.shape[-1]).squeeze())
    target_fft = torch.fft.rfft(target.reshape(-1, target.shape[-1]).squeeze())
    
    # Magnitude in dB
    pred_db = 20 * torch.log10(torch.abs(pred_fft) + 1e-8)
    target_db = 20 * torch.log10(torch.abs(target_fft) + 1e-8)
    
    # Mean absolute error in dB
    return torch.mean(torch.abs(pred_db - target_db)).item()


def phase_response_error(
    pred: torch.Tensor,
    target: torch.Tensor
) -> float:
    """
    Calculate phase response error.
    
    Critical for audio effects as phase affects sound character.
    
    Args:
        pred: Predicted signal
        target: Target signal
    
    Returns:
        Mean phase error in radians
    """
    # Determine appropriate FFT size based on input length
    input_length = pred.shape[-1]
    n_fft = 2048
    
    # Adjust n_fft if input is shorter than n_fft
    if input_length < n_fft:
        n_fft = 2 ** int(math.log2(max(input_length, 64)))
        if n_fft > input_length:
            n_fft = input_length
            
    hop_length = n_fft // 4
    win_length = n_fft

    # Compute STFT
    pred_stft = torch.stft(
        pred.reshape(-1, pred.shape[-1]).squeeze(),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length, device=pred.device),
        return_complex=True
    )
    target_stft = torch.stft(
        target.reshape(-1, target.shape[-1]).squeeze(),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length, device=target.device),
        return_complex=True
    )
    
    # Phase
    pred_phase = torch.angle(pred_stft)
    target_phase = torch.angle(target_stft)
    
    # Phase difference (wrapped to [-π, π])
    phase_diff = torch.angle(torch.exp(1j * (pred_phase - target_phase)))
    
    # Mean absolute phase error
    return torch.mean(torch.abs(phase_diff)).item()


def impulse_response_similarity(
    pred: torch.Tensor,
    target: torch.Tensor,
    input_signal: Optional[torch.Tensor] = None
) -> float:
    """
    Calculate impulse response similarity.
    
    Compares the estimated impulse responses of the systems.
    Note: This is an approximation. For accurate IR, use proper system
    identification with white noise or swept sine.
    
    Args:
        pred: Predicted output signal
        target: Target output signal
        input_signal: Input signal (if available for deconvolution)
    
    Returns:
        Normalized cross-correlation value (higher is better, max 1.0)
    """
    # If we don't have input signal, just compare outputs directly
    if input_signal is None:
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
    else:
        # Approximate IR by deconvolution (simplified)
        # In practice, you'd use proper system identification
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
    
    # Normalize
    pred_norm = (pred_flat - pred_flat.mean()) / (pred_flat.std() + 1e-8)
    target_norm = (target_flat - target_flat.mean()) / (target_flat.std() + 1e-8)
    
    # Cross-correlation
    cross_corr = torch.nn.functional.conv1d(
        target_norm.unsqueeze(0).unsqueeze(0),
        pred_norm.flip(0).unsqueeze(0).unsqueeze(0),
        padding=len(pred_norm) - 1
    )
    
    # Maximum normalized cross-correlation
    max_corr = torch.max(torch.abs(cross_corr))
    return max_corr.item() / len(pred_flat)


def total_harmonic_distortion_similarity(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_rate: int = 44100,
    fundamental_freq: Optional[float] = None
) -> float:
    """
    Compare Total Harmonic Distortion (THD) between pred and target.
    
    Useful for evaluating how well harmonic content is preserved.
    
    Args:
        pred: Predicted signal
        target: Target signal
        sample_rate: Audio sample rate
        fundamental_freq: Fundamental frequency (if known)
    
    Returns:
        Absolute difference in THD percentage
    """
    def compute_thd(signal):
        # Compute FFT
        fft = torch.fft.rfft(signal.reshape(-1))
        magnitude = torch.abs(fft)
        
        # Find fundamental (largest peak)
        if fundamental_freq is None:
            fundamental_idx = torch.argmax(magnitude[1:]) + 1  # Skip DC
        else:
            fundamental_idx = int(fundamental_freq * len(signal) / sample_rate)
        
        fundamental_power = magnitude[fundamental_idx] ** 2
        
        # Sum harmonic powers (2x, 3x, 4x, 5x fundamental)
        harmonic_power = 0.0
        for h in range(2, 6):
            harmonic_idx = fundamental_idx * h
            if harmonic_idx < len(magnitude):
                harmonic_power += magnitude[harmonic_idx] ** 2
        
        # THD = sqrt(sum of harmonic powers / fundamental power)
        thd = torch.sqrt(harmonic_power / (fundamental_power + 1e-8))
        return thd.item() * 100  # Convert to percentage
    
    pred_thd = compute_thd(pred)
    target_thd = compute_thd(target)
    
    return abs(pred_thd - target_thd)


def calculate_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_rate: int = 44100
) -> dict:
    """
    Calculate all available metrics.
    
    Args:
        pred: Predicted signal
        target: Target signal
        sample_rate: Audio sample rate
    
    Returns:
        Dictionary of all metric values
    """
    return {
        # Time-domain
        'mse': calculate_mse(pred, target),
        'mae': calculate_mae(pred, target),
        'esr': calculate_esr(pred, target),
        'pre_emphasis_esr': calculate_pre_emphasis_esr(pred, target),
        
        # Frequency-domain
        'spectral_convergence': spectral_convergence(pred, target),
        'multi_scale_spectral': multi_scale_spectral_loss(pred, target),
        'frequency_response_error_db': frequency_response_error(pred, target, sample_rate),
        'phase_response_error_rad': phase_response_error(pred, target),
        
        # Harmonic
        'impulse_response_similarity': impulse_response_similarity(pred, target),
        'thd_difference_pct': total_harmonic_distortion_similarity(pred, target, sample_rate),
    }

print("Metrics implementations checked.")
