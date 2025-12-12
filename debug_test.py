"""Quick diagnostic test for the failing test case."""
import torch
from src.training.loss import MultiScaleSpectralLoss

print("Testing small input fallback...")
loss_fn = MultiScaleSpectralLoss(fft_sizes=(2048, 1024, 512, 256, 128, 64))

# Very small input (smaller than smallest FFT size)
pred = torch.randn(1, 1, 32)
target = torch.randn(1, 1, 32)

print(f"Input shape: {pred.shape}")
print(f"Input length: {pred.shape[-1]}")

try:
    loss = loss_fn(pred, target)
    print(f"SUCCESS: Loss = {loss.item():.6f}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
