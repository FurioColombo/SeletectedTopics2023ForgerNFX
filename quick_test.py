"""Simple inline test for STFT fix"""
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.training.loss import MultiScaleSpectralLoss

# Test exact error condition from traceback: [1, 16, 512]
pred = torch.randn(1, 16, 512)
target = torch.randn(1, 16, 512)

print(f"Testing with shape: {pred.shape}")

loss_fn = MultiScaleSpectralLoss(fft_sizes=(2048, 1024, 512, 256, 128, 64))

try:
    loss = loss_fn(pred, target)
    print(f"SUCCESS - Loss computed: {loss.item():.6f}")
    print("The STFT padding error has been fixed!")
except RuntimeError as e:
    print(f"FAILED - Error still occurs: {e}")
    sys.exit(1)
