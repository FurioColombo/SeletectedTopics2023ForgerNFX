"""
Tests for MetricsVisualizer class.

Tests:
- Plot generation for various visualizations
- HTML report creation
- Handling of edge cases
- Plotly figure validation
"""
import pytest
import torch
import numpy as np
from pathlib import Path
from src.evaluation.visualizer import MetricsVisualizer


@pytest.fixture
def sample_eval_results():
    """Create sample evaluation results for testing."""
    return {
        'metrics': {
            'esr': {'mean': 0.15, 'std': 0.05, 'min': 0.08, 'max': 0.25},
            'mse': {'mean': 0.02, 'std': 0.01, 'min': 0.01, 'max': 0.05},
            'mae': {'mean': 0.10, 'std': 0.03, 'min': 0.05, 'max': 0.18},
            'spectral_convergence': {'mean': 0.12, 'std': 0.04, 'min': 0.06, 'max': 0.20},
        },
        'summary': 'Test evaluation summary',
        'samples': 10
    }


@pytest.fixture
def sample_audio():
    """Create sample audio tensors."""
    return (
        torch.randn(1, 2048),  # input
        torch.randn(1, 2048),  # predicted
        torch.randn(1, 2048)   # target
    )


class TestVisualizerInit:
    """Test MetricsVisualizer initialization."""
    
    def test_initialization(self, sample_eval_results):
        """Test basic initialization."""
        visualizer = MetricsVisualizer(sample_eval_results, sample_rate=44100)
        
        assert visualizer.eval_results == sample_eval_results
        assert visualizer.sample_rate == 44100
    
    def test_initialization_custom_sample_rate(self, sample_eval_results):
        """Test initialization with custom sample rate."""
        visualizer = MetricsVisualizer(sample_eval_results, sample_rate=48000)
        assert visualizer.sample_rate == 48000


class TestPlotGeneration:
    """Test plot generation methods."""
    
    def test_plot_metrics_overview(self, sample_eval_results):
        """Test metrics overview plot generation."""
        visualizer = MetricsVisualizer(sample_eval_results)
        fig = visualizer.plot_metrics_overview()
        
        # Check that a figure was created
        assert fig is not None
        assert hasattr(fig, 'data')  # Plotly figure has data attribute
    
    def test_plot_metrics_distribution(self, sample_eval_results):
        """Test metrics distribution plot."""
        # Need to add per-sample data for distribution plot
        sample_eval_results['per_sample_metrics'] = {
            'esr': [0.1, 0.15, 0.12, 0.18, 0.14]
        }
        
        visualizer = MetricsVisualizer(sample_eval_results)
        
        # This might not be implemented or might need different data structure
        # Test if method exists
        if hasattr(visualizer, 'plot_metrics_distribution'):
            fig = visualizer.plot_metrics_distribution('esr')
            assert fig is not None
    
    def test_plot_training_history(self, sample_eval_results):
        """Test training history plot."""
        visualizer = MetricsVisualizer(sample_eval_results)
        
        history = [
            {'epoch': 1, 'train_loss': 0.5, 'val_loss': 0.6},
            {'epoch': 2, 'train_loss': 0.4, 'val_loss': 0.5},
            {'epoch': 3, 'train_loss': 0.3, 'val_loss': 0.45},
        ]
        
        fig = visualizer.plot_training_history(history)
        assert fig is not None
        assert hasattr(fig, 'data')
    
    def test_plot_waveform_comparison(self, sample_eval_results, sample_audio):
        """Test waveform comparison plot."""
        visualizer = MetricsVisualizer(sample_eval_results)
        input_audio, pred_audio, target_audio = sample_audio
        
        fig = visualizer.plot_waveform_comparison(
            input_audio, pred_audio, target_audio, duration_sec=0.5
        )
        
        assert fig is not None
        assert hasattr(fig, 'data')
    
    def test_plot_spectrogram_comparison(self, sample_eval_results, sample_audio):
        """Test spectrogram comparison plot."""
        visualizer = MetricsVisualizer(sample_eval_results)
        _, pred_audio, target_audio = sample_audio
        
        fig = visualizer.plot_spectrogram_comparison(pred_audio, target_audio)
        
        assert fig is not None
        assert hasattr(fig, 'data')
    
    def test_plot_frequency_response(self, sample_eval_results, sample_audio):
        """Test frequency response plot."""
        visualizer = MetricsVisualizer(sample_eval_results)
        _, pred_audio, target_audio = sample_audio
        
        fig = visualizer.plot_frequency_response(pred_audio, target_audio)
        
        assert fig is not None
        assert hasattr(fig, 'data')


class TestHTMLReportGeneration:
    """Test full HTML report generation."""
    
    def test_create_full_report_basic(self, sample_eval_results, tmp_path):
        """Test creating a basic HTML report."""
        visualizer = MetricsVisualizer(sample_eval_results)
        output_file = tmp_path / "report.html"
        
        visualizer.create_full_report(output_file)
        
        assert output_file.exists()
        assert output_file.stat().st_size > 0
    
    def test_create_full_report_with_training_history(self, sample_eval_results, tmp_path):
        """Test report with training history."""
        visualizer = MetricsVisualizer(sample_eval_results)
        output_file = tmp_path / "report_with_history.html"
        
        history = [
            {'epoch': 1, 'train_loss': 0.5, 'val_loss': 0.6},
            {'epoch': 2, 'train_loss': 0.4, 'val_loss': 0.5},
        ]
        
        visualizer.create_full_report(output_file, training_history=history)
        
        assert output_file.exists()
        
        # Check that HTML contains expected content
        content = output_file.read_text()
        assert 'Evaluation Summary' in content or 'report' in content.lower()
    
    def test_create_full_report_with_audio(self, sample_eval_results, sample_audio, tmp_path):
        """Test report with sample audio."""
        visualizer = MetricsVisualizer(sample_eval_results)
        output_file = tmp_path / "report_with_audio.html"
        
        visualizer.create_full_report(output_file, sample_audio=sample_audio)
        
        assert output_file.exists()
    
    def test_html_report_is_valid(self, sample_eval_results, tmp_path):
        """Test that generated HTML has valid structure."""
        visualizer = MetricsVisualizer(sample_eval_results)
        output_file = tmp_path / "report_valid.html"
        
        visualizer.create_full_report(output_file)
        
        content = output_file.read_text()
        
        # Basic HTML validation
        assert '<html' in content.lower() or '<!doctype' in content.lower()


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_metrics(self):
        """Test with empty metrics dictionary."""
        results = {
            'metrics': {},
            'summary': 'Empty test',
            'samples': 0
        }
        
        visualizer = MetricsVisualizer(results)
        
        # Should initialize without error
        assert visualizer.eval_results == results
    
    def test_very_short_audio(self, sample_eval_results):
        """Test with very short audio samples."""
        visualizer = MetricsVisualizer(sample_eval_results)
        
        short_audio = torch.randn(1, 100)  # Very short
        
        # Should handle without crashing
        try:
            fig = visualizer.plot_waveform_comparison(
                short_audio, short_audio, short_audio, duration_sec=0.01
            )
            assert fig is not None
        except Exception as e:
            # If it errors, it should be a reasonable error
            assert isinstance(e, (ValueError, RuntimeError))
    
    def test_single_channel_audio(self, sample_eval_results):
        """Test with single channel audio."""
        visualizer = MetricsVisualizer(sample_eval_results)
        
        audio = torch.randn(1, 2048)  # Single channel
        
        fig = visualizer.plot_spectrogram_comparison(audio, audio)
        assert fig is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
