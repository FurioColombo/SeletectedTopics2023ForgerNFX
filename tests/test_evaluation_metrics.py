"""
Tests for evaluation metrics.
"""
import pytest
import torch
from src.evaluation.metrics import (
    calculate_esr,
    calculate_mae,
    spectral_convergence,
    phase_response_error,
    impulse_response_similarity,
    calculate_all_metrics
)


def test_esr():
    """Test ESR calculation."""
    pred = torch.randn(1, 1000)
    target = torch.randn(1, 1000)
    
    esr = calculate_esr(pred, target)
    assert isinstance(esr, float)
    assert esr >= 0.0


def test_mae():
    """Test MAE calculation."""
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.5, 2.5, 3.5])
    
    mae = calculate_mae(pred, target)
    assert abs(mae - 0.5) < 1e-6


def test_spectral_convergence():
    """Test spectral convergence metric."""
    # Create signals with known frequency content
    t = torch.linspace(0, 1, 44100)
    pred = torch.sin(2 * torch.pi * 440 * t)  # 440 Hz
    target = torch.sin(2 * torch.pi * 440 * t)  # Same
    
    sc = spectral_convergence(pred, target)
    assert isinstance(sc, float)
    assert sc < 0.1  # Should be very similar


def test_phase_response_error():
    """Test phase response error."""
    pred = torch.randn(1, 8192)
    target = torch.randn(1, 8192)
    
    phase_err = phase_response_error(pred, target)
    assert isinstance(phase_err, float)
    assert 0.0 <= phase_err <= torch.pi


def test_impulse_response_similarity():
    """Test impulse response similarity."""
    # Identical signals should have high similarity
    signal = torch.randn(1, 4096)
    
    similarity = impulse_response_similarity(signal, signal)
    assert isinstance(similarity, float)
    assert similarity > 0.5  # Should be reasonably high for identical


def test_calculate_all_metrics():
    """Test calculating all metrics at once."""
    pred = torch.randn(1, 8192)
    target = torch.randn(1, 8192)
    
    metrics = calculate_all_metrics(pred, target, sample_rate=44100)
    
    # Check all expected metrics are present
    expected_metrics = [
        'mse', 'mae', 'esr', 'pre_emphasis_esr',
        'spectral_convergence', 'multi_scale_spectral',
        'frequency_response_error_db', 'phase_response_error_rad',
        'impulse_response_similarity', 'thd_difference_pct'
    ]
    
    for metric in expected_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], float)


def test_metrics_with_identical_signals():
    """Test that identical signals give near-zero error."""
    signal = torch.randn(1, 4096)
    
    metrics = calculate_all_metrics(signal, signal)
    
    # MSE and MAE should be exactly 0
    assert metrics['mse'] < 1e-10
    assert metrics['mae'] < 1e-10
    
    # ESR should be near 0
    assert metrics['esr'] < 1e-6


def test_metrics_with_different_signals():
    """Test that different signals give non-zero error."""
    pred = torch.randn(1, 4096)
    target = torch.randn(1, 4096)
    
    metrics = calculate_all_metrics(pred, target)
    
    # All error metrics should be non-zero
    assert metrics['mse'] > 0
    assert metrics['mae'] > 0
    assert metrics['esr'] > 0
