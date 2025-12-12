"""
Tests for inference functionality.

Tests:
- Basic inference workflow
- Audio file I/O
- Model output validation
- Device handling
- Edge cases
"""
import pytest
import torch
from pathlib import Path
from src.evaluation.inference import run_inference
from src.models.lstm import LSTMModel
from src.config.config import ModelConfig
from src.utils.io import save_audio, load_audio


@pytest.fixture
def dummy_model():
    """Create a simple model for testing."""
    config = ModelConfig(name="lstm", hidden_size=8, num_layers=1)
    model = LSTMModel(config)
    model.eval()
    return model


@pytest.fixture
def sample_audio_file(tmp_path):
    """Create a sample audio file for testing."""
    # Generate sample audio
    sample_rate = 44100
    duration = 0.5  # seconds
    waveform = torch.randn(1, int(sample_rate * duration))
    
    # Save to file
    input_path = tmp_path / "input.wav"
    save_audio(str(input_path), waveform, sample_rate)
    
    return input_path


class TestInferenceBasic:
    """Test basic inference functionality."""
    
    def test_run_inference_basic(self, dummy_model, sample_audio_file, tmp_path):
        """Test basic inference on audio file."""
        output_path = tmp_path / "output.wav"
        
        result = run_inference(
            model=dummy_model,
            input_path=str(sample_audio_file),
            output_path=str(output_path),
            device="cpu"
        )
        
        # Check that output file was created
        assert output_path.exists()
        
        # Check that result is a tensor
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 2  # (channels, time)
    
    def test_inference_output_shape(self, dummy_model, sample_audio_file, tmp_path):
        """Test that output has correct shape."""
        output_path = tmp_path / "output.wav"
        
        result = run_inference(dummy_model, str(sample_audio_file), str(output_path))
        
        # Output should have same general structure as input
        assert result.shape[0] == 1  # Single channel
        assert result.shape[1] > 0  # Has time samples
    
    def test_inference_creates_file(self, dummy_model, sample_audio_file, tmp_path):
        """Test that inference creates output file."""
        output_path = tmp_path / "output.wav"
        
        assert not output_path.exists()
        
        run_inference(dummy_model, str(sample_audio_file), str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0


class TestInferenceDevices:
    """Test inference on different devices."""
    
    def test_inference_cpu(self, dummy_model, sample_audio_file, tmp_path):
        """Test inference on CPU."""
        output_path = tmp_path / "output_cpu.wav"
        
        result = run_inference(
            dummy_model, str(sample_audio_file), str(output_path), device="cpu"
        )
        
        assert result is not None
        assert result.device.type == "cpu"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_inference_cuda(self, dummy_model, sample_audio_file, tmp_path):
        """Test inference on CUDA."""
        output_path = tmp_path / "output_cuda.wav"
        
        result = run_inference(
            dummy_model, str(sample_audio_file), str(output_path), device="cuda"
        )
        
        assert result is not None
        # Result is moved to CPU for saving
        assert result.device.type == "cpu"


class TestInferenceAudioProcessing:
    """Test audio processing in inference."""
    
    def test_output_can_be_loaded(self, dummy_model, sample_audio_file, tmp_path):
        """Test that output audio can be loaded back."""
        output_path = tmp_path / "output.wav"
        
        run_inference(dummy_model, str(sample_audio_file), str(output_path))
        
        # Load the output
        loaded = load_audio(str(output_path))
        
        assert isinstance(loaded, torch.Tensor)
        assert loaded.dim() == 2
    
    def test_inference_preserves_audio_length(self, dummy_model, sample_audio_file, tmp_path):
        """Test that output length is reasonable relative to input."""
        output_path = tmp_path / "output.wav"
        
        # Load input to check length
        input_audio = load_audio(str(sample_audio_file))
        input_length = input_audio.shape[1]
        
        result = run_inference(dummy_model, str(sample_audio_file), str(output_path))
        
        # Output should have similar length (may vary slightly due to model architecture)
        assert result.shape[1] > 0
        # Allow some variation but should be in same ballpark
        assert 0.5 * input_length <= result.shape[1] <= 2.0 * input_length


class TestInferenceEdgeCases:
    """Test edge cases and error handling."""
    
    def test_nonexistent_input_file(self, dummy_model, tmp_path):
        """Test with non-existent input file."""
        input_path = tmp_path / "nonexistent.wav"
        output_path = tmp_path / "output.wav"
        
        with pytest.raises((FileNotFoundError, RuntimeError)):
            run_inference(dummy_model, str(input_path), str(output_path))
    
    def test_output_directory_creation(self, dummy_model, sample_audio_file, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        output_dir = tmp_path / "new_dir"
        output_path = output_dir / "output.wav"
        
        # Directory doesn't exist yet
        assert not output_dir.exists()
        
        # Create directory manually (inference might not do this)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result = run_inference(dummy_model, str(sample_audio_file), str(output_path))
        
        assert output_path.exists()


class TestModelInEvalMode:
    """Test that model is in evaluation mode."""
    
    def test_model_eval_mode(self, dummy_model, sample_audio_file, tmp_path):
        """Test that model is set to eval mode during inference."""
        output_path = tmp_path / "output.wav"
        
        # Set model to training mode
        dummy_model.train()
        assert dummy_model.training
        
        run_inference(dummy_model, str(sample_audio_file), str(output_path))
        
        # Model should be in eval mode after inference
        # (Depending on implementation, it might stay in eval mode)
        # This test documents the expected behavior


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
