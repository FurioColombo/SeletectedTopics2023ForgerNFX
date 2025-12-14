"""
Comprehensive tests for ModelEvaluator class.

Tests:
- Evaluation workflow with different data loaders
- Metric aggregation (mean, std, min, max)
- Results saving (JSON, CSV formats)
- Audio prediction saving
- Model comparison functionality
- Edge cases and error handling
"""
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import json
import csv

from src.evaluation.evaluator import ModelEvaluator
from src.models.lstm import LSTMModel
from src.config.config import ModelConfig


@pytest.fixture
def dummy_model():
    """Create a simple model for testing."""
    config = ModelConfig(name="lstm", hidden_size=8, num_layers=1)
    model = LSTMModel(config)
    return model


@pytest.fixture
def test_dataloader():
    """Create a simple test data loader."""
    # Small dataset for fast testing
    # Increase input size to avoid potential STFT issues in metrics
    inputs = torch.randn(10, 1, 2048)  # 10 samples, 1 channel, 2048 samples
    targets = torch.randn(10, 1, 2048)
    
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    return loader


class TestModelEvaluatorInit:
    """Test ModelEvaluator initialization."""
    
    def test_initialization(self, dummy_model, test_dataloader):
        """Test basic initialization."""
        evaluator = ModelEvaluator(
            model=dummy_model,
            test_loader=test_dataloader,
            device="cpu",
            sample_rate=44100
        )
        
        assert evaluator.model is dummy_model
        assert evaluator.test_loader is test_dataloader
        assert evaluator.device == "cpu"
        assert evaluator.sample_rate == 44100
    
    def test_model_set_to_eval_mode(self, dummy_model, test_dataloader):
        """Model should be in eval mode after initialization."""
        evaluator = ModelEvaluator(dummy_model, test_dataloader)
        # Note: Model is set to eval during evaluate(), not init, 
        # but the class currently sets it in __init__ too.
        assert not dummy_model.training


class TestModelEvaluatorEvaluate:
    """Test the evaluate() method."""
    
    def test_evaluate_returns_dict(self, dummy_model, test_dataloader):
        """evaluate() should return a dictionary with expected keys."""
        evaluator = ModelEvaluator(dummy_model, test_dataloader)
        results = evaluator.evaluate(save_predictions=False)
        
        assert isinstance(results, dict)
        assert 'metrics' in results
        assert 'summary' in results
        assert 'per_sample_metrics' in results
    
    def test_evaluate_metrics_structure(self, dummy_model, test_dataloader):
        """Verify metrics dictionary structure."""
        evaluator = ModelEvaluator(dummy_model, test_dataloader)
        results = evaluator.evaluate(save_predictions=False)
        
        metrics = results['metrics']
        
        # Check that metrics have mean, std, min, max
        for metric_name, metric_data in metrics.items():
            assert isinstance(metric_data, dict)
            assert 'mean' in metric_data
            assert 'std' in metric_data
            assert 'min' in metric_data
            assert 'max' in metric_data
    
    def test_evaluate_no_save_predictions(self, dummy_model, test_dataloader):
        """Test evaluation without saving predictions."""
        evaluator = ModelEvaluator(dummy_model, test_dataloader)
        results = evaluator.evaluate(save_predictions=False)
        
        # Check per_sample_metrics is present
        assert len(results['per_sample_metrics']) == 10  # 10 samples in dataloader
    
    def test_evaluate_with_save_predictions(self, dummy_model, test_dataloader, tmp_path):
        """Test evaluation with saving predictions."""
        evaluator = ModelEvaluator(dummy_model, test_dataloader)
        output_dir = tmp_path / "predictions"
        
        results = evaluator.evaluate(save_predictions=True, output_dir=output_dir)
        
        assert len(results['per_sample_metrics']) == 10
        assert output_dir.exists()
        
        # Check that WAV files were created
        wav_files = list(output_dir.glob("*.wav"))
        assert len(wav_files) > 0
    
    def test_evaluate_summary_is_string(self, dummy_model, test_dataloader):
        """Summary should be a formatted string."""
        evaluator = ModelEvaluator(dummy_model, test_dataloader)
        results = evaluator.evaluate(save_predictions=False)
        
        assert isinstance(results['summary'], str)
        assert len(results['summary']) > 0
        assert "SUMMARY" in results['summary']


class TestMetricAggregation:
    """Test metric aggregation functionality."""
    
    def test_aggregation_statistics(self, dummy_model, test_dataloader):
        """Test that statistics are correctly computed."""
        evaluator = ModelEvaluator(dummy_model, test_dataloader)
        results = evaluator.evaluate(save_predictions=False)
        
        # Check a specific metric (e.g., ESR)
        esr_stats = results['metrics']['esr']
        
        # Mean should be between min and max
        assert esr_stats['min'] <= esr_stats['mean'] <= esr_stats['max']
        
        # Std should be non-negative
        assert esr_stats['std'] >= 0
    
    def test_all_metrics_present(self, dummy_model, test_dataloader):
        """Verify all expected metrics are in results."""
        evaluator = ModelEvaluator(dummy_model, test_dataloader)
        results = evaluator.evaluate(save_predictions=False)
        
        # FIX: Updated keys to match metrics.py implementation
        expected_metrics = [
            'esr', 'mse', 'mae', 'pre_emphasis_esr',
            'spectral_convergence', 'multi_scale_spectral', # not _loss
            'frequency_response_error_db', 'phase_response_error_rad',
            'impulse_response_similarity', 'thd_difference_pct'
        ]
        
        for metric in expected_metrics:
            assert metric in results['metrics'], f"Missing metric: {metric}"


class TestResultsSaving:
    """Test save_results() functionality."""
    
    def test_save_json(self, dummy_model, test_dataloader, tmp_path):
        """Test saving results as JSON."""
        evaluator = ModelEvaluator(dummy_model, test_dataloader)
        results = evaluator.evaluate(save_predictions=False)
        
        output_file = tmp_path / "results.json"
        evaluator.save_results(results, output_file, format='json')
        
        assert output_file.exists()
        
        # Verify JSON is valid and contains expected data
        with open(output_file, 'r') as f:
            loaded_results = json.load(f)
        
        assert 'metrics' in loaded_results
        assert 'summary' in loaded_results
    
    def test_save_csv(self, dummy_model, test_dataloader, tmp_path):
        """Test saving results as CSV."""
        evaluator = ModelEvaluator(dummy_model, test_dataloader)
        results = evaluator.evaluate(save_predictions=False)
        
        output_file = tmp_path / "results.csv"
        evaluator.save_results(results, output_file, format='csv')
        
        # Check for generated files (suffix added by save_results)
        aggregated_path = output_file.with_suffix('.aggregated.csv')
        assert aggregated_path.exists()
        
        # Verify CSV has correct structure
        with open(aggregated_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) > 0
        assert 'Metric' in rows[0]
        assert 'Mean' in rows[0]


class TestModelComparison:
    """Test compare_models() functionality."""
    
    def test_compare_models_basic(self, dummy_model, test_dataloader):
        """Test basic model comparison."""
        evaluator1 = ModelEvaluator(dummy_model, test_dataloader)
        evaluator2 = ModelEvaluator(dummy_model, test_dataloader)
        
        comparison = evaluator1.compare_models(evaluator2, metric_name='esr')
        
        assert isinstance(comparison, dict)
        assert 'improvement_pct' in comparison
    
    def test_compare_different_metric(self, dummy_model, test_dataloader):
        """Test comparison with different metrics."""
        evaluator1 = ModelEvaluator(dummy_model, test_dataloader)
        evaluator2 = ModelEvaluator(dummy_model, test_dataloader)
        
        for metric in ['mse', 'mae', 'spectral_convergence']:
            comparison = evaluator1.compare_models(evaluator2, metric_name=metric)
            assert isinstance(comparison, dict)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataloader(self, dummy_model):
        """Test with empty dataloader (edge case)."""
        # Create empty dataset
        inputs = torch.empty(0, 1, 2048)
        targets = torch.empty(0, 1, 2048)
        dataset = TensorDataset(inputs, targets)
        loader = DataLoader(dataset, batch_size=2)
        
        evaluator = ModelEvaluator(dummy_model, loader)
        
        # evaluate() returns empty dicts for empty input
        results = evaluator.evaluate(save_predictions=False)
        assert results['metrics'] == {}
    
    def test_single_batch(self, dummy_model):
        """Test with single-batch dataloader."""
        inputs = torch.randn(2, 1, 2048)
        targets = torch.randn(2, 1, 2048)
        dataset = TensorDataset(inputs, targets)
        loader = DataLoader(dataset, batch_size=2)
        
        evaluator = ModelEvaluator(dummy_model, loader)
        results = evaluator.evaluate(save_predictions=False)
        
        assert 'metrics' in results
        assert 'esr' in results['metrics']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
