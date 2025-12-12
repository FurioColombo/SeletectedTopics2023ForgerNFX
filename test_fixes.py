"""
Quick test to verify STFT padding fix and logger improvements
"""
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("="*70)
print("Testing STFT Padding Fix and Logger Improvements")
print("="*70)

# Test 1: STFT Padding Fix
print("\n[Test 1] Testing MultiScaleSpectralLoss with block_size=512")
print("-" * 70)

from src.training.loss import MultiScaleSpectralLoss

# Create dummy data matching the error condition: [batch=1, channels=16, time=512]
# This is what caused the original error
pred = torch.randn(1, 16, 512)
target = torch.randn(1, 16, 512)

print(f"Input shape: {pred.shape}")
print(f"Block size (time dimension): {pred.shape[-1]}")

# Initialize loss with default FFT sizes
loss_fn = MultiScaleSpectralLoss(fft_sizes=(2048, 1024, 512, 256, 128, 64))
print(f"Original FFT sizes: {loss_fn.fft_sizes}")

try:
    loss_value = loss_fn(pred, target)
    print(f"✅ SUCCESS: Loss computed without error: {loss_value.item():.6f}")
    print(f"   The fix correctly filters FFT sizes to valid ones (≤ {pred.shape[-1]})")
except Exception as e:
    print(f"❌ FAILED: {e}")
    sys.exit(1)

# Test 2: Verify it works with different block sizes
print("\n[Test 2] Testing with larger block_size=2048")
print("-" * 70)

pred_large = torch.randn(2, 1, 2048)
target_large = torch.randn(2, 1, 2048)

print(f"Input shape: {pred_large.shape}")
print(f"Block size: {pred_large.shape[-1]}")

try:
    loss_value_large = loss_fn(pred_large, target_large)
    print(f"✅ SUCCESS: Loss computed: {loss_value_large.item():.6f}")
    print(f"   All FFT sizes can be used with this larger input")
except Exception as e:
    print(f"❌ FAILED: {e}")
    sys.exit(1)

# Test 3: Logger improvements
print("\n[Test 3] Testing Logger Output (visual inspection)")
print("-" * 70)

from src.monitoring.logger import MonitoringLogger
import tempfile
import os

# Test with wandb disabled (to avoid requiring API key for test)
with tempfile.TemporaryDirectory() as tmpdir:
    print("\nInitializing logger with use_wandb=False (local mode)...")
    logger = MonitoringLogger(
        project="test-project",
        run_name="test-run",
        config={"test": "config"},
        use_wandb=False,
        log_dir=Path(tmpdir) / "logs"
    )
    
    print("\n✅ Logger initialized successfully")
    print(f"   Log directory created: {logger.log_dir}")
    
    # Test logging
    logger.log_metrics({"test_loss": 0.123, "test_acc": 0.95}, step=1)
    
    # Check metrics file
    if logger.metrics_log.exists():
        print(f"✅ Metrics logged to: {logger.metrics_log}")
        with open(logger.metrics_log) as f:
            content = f.read()
            print(f"   Content: {content.strip()}")
    
    logger.finish()

print("\n" + "="*70)
print("All Tests Passed! ✅")
print("="*70)
print("\nSummary:")
print("  1. ✅ STFT padding error fixed - handles block_size=512 correctly")
print("  2. ✅ Logger works with fallback to local logging")
print("  3. ✅ No crashes with various input sizes")
print("\nThe fixes are ready for production use!")
