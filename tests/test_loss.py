"""
Tests for training loss functions.

Tests cover:
- MultiScaleSpectralLoss with various input sizes
- STFT padding error prevention (regression test)
- MSELoss and ESRLoss
- CombinedLoss functionality
"""
import pytest
import torch
from src.training.loss import (
    MSELoss, 
    ESRLoss, 
    MultiScaleSpectralLoss, 
    CombinedLoss
)


class TestMSELoss:
    """Test MSELoss implementation."""
    
    def test_identical_signals(self):
        """MSE should be zero for identical signals."""
        loss_fn = MSELoss()
        pred = torch.ones(2, 1, 100)
        target = torch.ones(2, 1, 100)
        
        loss = loss_fn(pred, target)
        assert loss.item() < 1e-6
    
    def test_different_signals(self):
        """MSE should be positive for different signals."""
        loss_fn = MSELoss()
        pred = torch.zeros(2, 1, 100)
        target = torch.ones(2, 1, 100)
        
        loss = loss_fn(pred, target)
        assert loss.item() > 0.9


class TestESRLoss:
    """Test Error Signal Ratio Loss."""
    
    def test_identical_signals(self):
        """ESR should be zero for identical signals."""
        loss_fn = ESRLoss()
        pred = torch.randn(2, 1, 100)
        target = pred.clone()
        
        loss = loss_fn(pred, target)
        assert loss.item() < 1e-6
    
    def test_zero_prediction(self):
        """ESR should be ~1.0 when prediction is all zeros."""
        loss_fn = ESRLoss()
        pred = torch.zeros(2, 1, 100)
        target = torch.ones(2, 1, 100)
        
        loss = loss_fn(pred, target)
        assert 0.9 < loss.item() < 1.1


class TestMultiScaleSpectralLoss:
    """Test Multi-Scale Spectral Loss with emphasis on STFT padding fix."""
    
    def test_block_size_512_regression(self):
        """
        Regression test for STFT padding error with block_size=512.
        
        This was the original error condition:
        RuntimeError: Argument #4: Padding size should be less than the 
        corresponding input dimension, but got: padding (1024, 1024) 
        at dimension 2 of input [1, 16, 512]
        """
        loss_fn = MultiScaleSpectralLoss(fft_sizes=(2048, 1024, 512, 256, 128, 64))
        
        # Exact shape from the error traceback
        pred = torch.randn(1, 16, 512)
        target = torch.randn(1, 16, 512)
        
        # Should not raise RuntimeError
        loss = loss_fn(pred, target)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
    
    def test_various_block_sizes(self):
        """Test that loss works with various input sizes."""
        loss_fn = MultiScaleSpectralLoss(fft_sizes=(2048, 1024, 512, 256, 128, 64))
        
        block_sizes = [256, 512, 1024, 2048, 4096]
        
        for block_size in block_sizes:
            pred = torch.randn(2, 1, block_size)
            target = torch.randn(2, 1, block_size)
            
            loss = loss_fn(pred, target)
            assert isinstance(loss, torch.Tensor)
            assert loss.item() >= 0, f"Loss should be non-negative for block_size={block_size}"
    
    def test_small_input_fallback(self):
        """Test fallback behavior for very small inputs."""
        loss_fn = MultiScaleSpectralLoss(fft_sizes=(2048, 1024, 512, 256, 128, 64))
        
        # Very small input (smaller than smallest FFT size)
        pred = torch.randn(1, 1, 32)
        target = torch.randn(1, 1, 32)
        
        # Should use fallback FFT size
        loss = loss_fn(pred, target)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
    
    def test_identical_signals_low_loss(self):
        """Identical signals should have low spectral loss."""
        loss_fn = MultiScaleSpectralLoss(fft_sizes=(512, 256, 128, 64))
        
        signal = torch.randn(2, 1, 1024)
        
        loss = loss_fn(signal, signal.clone())
        assert loss.item() < 1e-4
    
    def test_batch_processing(self):
        """Test that loss works with various batch sizes."""
        loss_fn = MultiScaleSpectralLoss(fft_sizes=(512, 256, 128, 64))
        
        for batch_size in [1, 4, 8, 16]:
            pred = torch.randn(batch_size, 1, 1024)
            target = torch.randn(batch_size, 1, 1024)
            
            loss = loss_fn(pred, target)
            assert isinstance(loss, torch.Tensor)
            assert loss.item() >= 0


class TestCombinedLoss:
    """Test CombinedLoss with multiple loss functions."""
    
    def test_single_loss(self):
        """CombinedLoss with single loss should match that loss."""
        mse = MSELoss()
        combined = CombinedLoss({mse: 1.0})
        
        pred = torch.randn(2, 1, 100)
        target = torch.randn(2, 1, 100)
        
        mse_loss = mse(pred, target)
        combined_loss = combined(pred, target)
        
        assert torch.allclose(mse_loss, combined_loss)
    
    def test_weighted_combination(self):
        """Test that weights are applied correctly."""
        mse = MSELoss()
        esr = ESRLoss()
        
        pred = torch.randn(2, 1, 100)
        target = torch.randn(2, 1, 100)
        
        mse_loss = mse(pred, target)
        esr_loss = esr(pred, target)
        
        # Test different weight combinations
        combined1 = CombinedLoss({mse: 1.0, esr: 0.5})
        loss1 = combined1(pred, target)
        expected1 = mse_loss + 0.5 * esr_loss
        
        assert torch.allclose(loss1, expected1)
    
    def test_with_spectral_loss(self):
        """Test combination including spectral loss."""
        mse = MSELoss()
        spectral = MultiScaleSpectralLoss(fft_sizes=(512, 256, 128))
        
        combined = CombinedLoss({mse: 1.0, spectral: 0.1})
        
        pred = torch.randn(2, 1, 1024)
        target = torch.randn(2, 1, 1024)
        
        loss = combined(pred, target)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


class TestLossGradients:
    """Test that losses produce valid gradients for training."""
    
    def test_mse_gradients(self):
        """MSELoss should produce valid gradients."""
        loss_fn = MSELoss()
        pred = torch.randn(2, 1, 100, requires_grad=True)
        target = torch.randn(2, 1, 100)
        
        loss = loss_fn(pred, target)
        loss.backward()
        
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()
    
    def test_spectral_gradients(self):
        """MultiScaleSpectralLoss should produce valid gradients."""
        loss_fn = MultiScaleSpectralLoss(fft_sizes=(512, 256, 128))
        pred = torch.randn(2, 1, 1024, requires_grad=True)
        target = torch.randn(2, 1, 1024)
        
        loss = loss_fn(pred, target)
        loss.backward()
        
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
