"""
GPU utilities for dynamic resource management.
"""
import torch
from typing import Callable, Any


def find_optimal_batch_size(
    test_fn: Callable[[int], Any],
    min_batch_size: int = 1,
    max_batch_size: int = 512,
    device: str = "cuda"
) -> int:
    """
    Find the largest batch size that fits in GPU memory using binary search.
    
    Args:
        test_fn: Function that takes batch_size and runs a test forward pass.
                 Should raise RuntimeError on OOM.
        min_batch_size: Minimum batch size to try
        max_batch_size: Maximum batch size to try
        device: Device to test on
        
    Returns:
        Optimal batch size that fits in memory
    """
    if not torch.cuda.is_available() or device == "cpu":
        return min_batch_size
    
    # Clear cache before testing
    torch.cuda.empty_cache()
    
    def can_fit(batch_size: int) -> bool:
        """Test if a batch size fits in memory."""
        try:
            torch.cuda.empty_cache()
            test_fn(batch_size)
            torch.cuda.empty_cache()
            return True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return False
            raise  # Re-raise if it's not an OOM error
    
    # Binary search for optimal batch size
    left, right = min_batch_size, max_batch_size
    optimal = min_batch_size
    
    while left <= right:
        mid = (left + right) // 2
        
        if can_fit(mid):
            optimal = mid
            left = mid + 1  # Try larger
        else:
            right = mid - 1  # Try smaller
    
    print(f"🎯 Optimal batch size for {device}: {optimal}")
    return optimal


def get_gpu_memory_info() -> dict:
    """
    Get GPU memory information.
    
    Returns:
        Dictionary with memory stats (empty if no GPU)
    """
    if not torch.cuda.is_available():
        return {}
    
    return {
        'total_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
        'allocated_gb': torch.cuda.memory_allocated(0) / 1e9,
        'cached_gb': torch.cuda.memory_reserved(0) / 1e9,
        'free_gb': (torch.cuda.get_device_properties(0).total_memory - 
                    torch.cuda.memory_reserved(0)) / 1e9
    }


def print_gpu_info():
    """Print GPU information and memory stats."""
    if not torch.cuda.is_available():
        print("ℹ️  No GPU available, using CPU")
        return
    
    device_name = torch.cuda.get_device_name(0)
    mem_info = get_gpu_memory_info()
    
    print(f"🎮 GPU: {device_name}")
    print(f"   Total: {mem_info['total_gb']:.2f} GB")
    print(f"   Free: {mem_info['free_gb']:.2f} GB")
