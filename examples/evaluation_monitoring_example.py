"""
Example usage of the evaluation and monitoring system.

This script demonstrates how to use the new evaluation module and
live monitoring for comprehensive model assessment.
"""
import torch
from pathlib import Path

from src.config.paths import paths
from src.models.lstm import LSTMModel
from src.config.config import ModelConfig
from src.data.egfx import EGFxDataset
from src.data.loader import create_dataloaders
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.visualizer import MetricsVisualizer
from src.monitoring.logger import MonitoringLogger


def example_evaluation():
    """Example: Evaluate a trained model on test set."""
    print("=" * 60)
    print("EVALUATION EXAMPLE")
    print("=" * 60)
    
    # 1. Load model
    model_config = ModelConfig(name="lstm", hidden_size=64)
    model = LSTMModel(model_config)
    
    # Load trained weights (example path)
    checkpoint_path = paths.get_checkpoint_path("TubeScreamer_lstm")
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        print(f"✅ Loaded model from {checkpoint_path}")
    else:
        print(f"⚠️  No checkpoint found, using untrained model for demo")
    
    # 2. Load test data
    dataset = EGFxDataset(
        input_root=str(paths.DATASETS / "EGFxDataset" / "Clean"),
        output_root=str(paths.DATASETS / "EGFxDataset" / "TubeScreamer"),
        block_size=2048
    )
    
    from src.config.config import ProjectConfig
    config = ProjectConfig()
    train_loader, val_loader, test_loader = create_dataloaders(dataset, config)
    
    # 3. Run evaluation
    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        device="cpu",
        sample_rate=44100
    )
    
    print("\n🔍 Running evaluation on test set...")
    results = evaluator.evaluate(
        save_predictions=True,
        output_dir=paths.OUTPUTS / "evaluation" / "predictions"
    )
    
    # 4. Print summary
    print("\n" + results['summary'])
    
    # 5. Save results
    results_path = paths.OUTPUTS / "evaluation" / "results.json"
    evaluator.save_results(results, results_path, format='json')
    evaluator.save_results(results, results_path.with_suffix('.csv'), format='csv')
    
    print(f"\n💾 Results saved to {results_path.parent}")
    
    # 6. Create interactive visualizations
    visualizer = MetricsVisualizer(results, sample_rate=44100)
    
    # Get sample audio for visualization
    sample_batch = next(iter(test_loader))
    inputs, targets = sample_batch
    with torch.no_grad():
        predictions = model(inputs)
    
    # Create full HTML report
    report_path = paths.OUTPUTS / "evaluation" / "interactive_report.html"
    visualizer.create_full_report(
        output_path=report_path,
        sample_audio=(inputs[0], predictions[0], targets[0])
    )
    
    print(f"\n📊 Interactive report: {report_path}")
    print("   Open in browser to explore results!")
    
    return results


def example_monitoring():
    """Example: Use monitoring during training."""
    print("\n" + "=" * 60)
    print("MONITORING EXAMPLE")
    print("=" * 60)
    
    # Create monitoring logger
    logger = MonitoringLogger(
        project="forger-nfx",
        run_name="example_run",
        config={
            "model": "lstm",
            "hidden_size": 64,
            "epochs": 10,
            "batch_size": 32
        },
        use_wandb=True  # Set to False to test local-only logging
    )
    
    # Simulate training loop
    print("\n🏋️ Simulating training...")
    for epoch in range(3):
        # Simulate metrics
        train_loss = 0.05 - epoch * 0.01
        val_loss = 0.06 - epoch * 0.01
        
        logger.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch": epoch
        }, step=epoch)
        
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    
    # Finish logging
    logger.finish()
    
    print(f"\n✅ Monitoring demo complete")
    if logger.wandb_available:
        print(f"   View dashboard at wandb.ai")
    print(f"   Local logs at: {logger.log_dir}")


def example_kaggle_monitoring():
    """Example: Setup monitoring for Kaggle."""
    print("\n" + "=" * 60)
    print("KAGGLE MONITORING SETUP")
    print("=" * 60)
    
    print("""
To use wandb on Kaggle:

1. Get your wandb API key:
   - Go to https://wandb.ai/authorize
   - Copy your API key

2. Add to Kaggle secrets:
   - In your Kaggle notebook, go to Add-ons → Secrets
   - Click "+ Add a new secret"
   - Label: wandb_api_key
   - Value: <paste your API key>

3. In your Kaggle notebook code:
   ```python
   from src.monitoring import create_kaggle_logger
   
   logger = create_kaggle_logger(
       project="forger-nfx",
       run_name="kaggle_run_1",
       config={"model": "lstm", "epochs": 100}
   )
   
   # Use logger in training loop
   logger.log_metrics({"loss": loss_value}, step=epoch)
   ```

4. View your dashboard:
   - Dashboard URL will be printed when logger initializes
   - You can view live metrics from anywhere!
    """)


if __name__ == "__main__":
    # Run examples
    print("🎸 Forger NFX - Evaluation & Monitoring Examples\n")
    
    try:
        # Example 1: Full evaluation
        results = example_evaluation()
        
        # Example 2: Monitoring
        example_monitoring()
        
        # Example 3: Kaggle setup info
        example_kaggle_monitoring()
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("   This is expected if you don't have trained models or datasets yet")
        print("   The code structure is ready to use when you do!")
