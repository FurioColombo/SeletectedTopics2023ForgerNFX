"""
Interactive visualization for model evaluation results.

Uses Plotly to create modern, interactive HTML dashboards for analyzing
model performance, comparing outputs, and visualizing audio characteristics.
"""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class MetricsVisualizer:
    """
    Creates interactive visualizations for evaluation results.
    """
    
    def __init__(self, eval_results: Dict, sample_rate: int = 44100):
        """
        Initialize visualizer with evaluation results.
        
        Args:
            eval_results: Results dictionary from ModelEvaluator.evaluate()
            sample_rate: Audio sample rate
        """
        self.results = eval_results
        self.sample_rate = sample_rate
    
    def plot_metrics_overview(self) -> go.Figure:
        """
        Create overview plot of all metrics with error bars.
        
        Returns:
            Plotly figure
        """
        metrics = self.results['metrics']
        
        metric_names = list(metrics.keys())
        means = [metrics[m]['mean'] for m in metric_names]
        stds = [metrics[m]['std'] for m in metric_names]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Metrics',
            x=metric_names,
            y=means,
            error_y=dict(type='data', array=stds),
            marker_color='indianred',
            hovertemplate='<b>%{x}</b><br>Mean: %{y:.4f}<br>Std: %{customdata:.4f}<extra></extra>',
            customdata=stds
        ))
        
        fig.update_layout(
            title='Evaluation Metrics Overview',
            xaxis_title='Metric',
            yaxis_title='Value',
            template='plotly_white',
            height=500,
            xaxis={'tickangle': -45}
        )
        
        return fig
    
    def plot_metrics_distribution(self, metric_name: str) -> go.Figure:
        """
        Plot distribution of a specific metric across samples.
        
        Args:
            metric_name: Name of metric to plot
        
        Returns:
            Plotly figure
        """
        per_sample = self.results['per_sample_metrics']
        values = [s[metric_name] for s in per_sample]
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=values,
            nbinsx=30,
            name=metric_name,
            marker_color='steelblue'
        ))
        
        # Add mean line
        mean_val = np.mean(values)
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_val:.4f}"
        )
        
        fig.update_layout(
            title=f'Distribution of {metric_name}',
            xaxis_title=metric_name,
            yaxis_title='Count',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def plot_training_history(self, history: List[Dict]) -> go.Figure:
        """
        Plot training loss curves.
        
        Args:
            history: Training history (list of dicts with 'epoch', 'train_loss', 'val_loss')
        
        Returns:
            Plotly figure
        """
        epochs = [h['epoch'] for h in history]
        train_loss = [h['train_loss'] for h in history]
        val_loss = [h['val_loss'] for h in history]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=epochs,
            y=train_loss,
            mode='lines+markers',
            name='Training Loss',
            line=dict(color='royalblue', width=2),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=epochs,
            y=val_loss,
            mode='lines+markers',
            name='Validation Loss',
            line=dict(color='crimson', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title='Training History',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            template='plotly_white',
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def plot_waveform_comparison(
        self,
        input_audio: torch.Tensor,
        predicted_audio: torch.Tensor,
        target_audio: torch.Tensor,
        duration_sec: float = 1.0
    ) -> go.Figure:
        """
        Plot waveform comparison (input, predicted, target).
        
        Args:
            input_audio: Input waveform
            predicted_audio: Predicted output waveform
            target_audio: Target waveform
            duration_sec: Duration to plot (seconds)
        
        Returns:
            Plotly figure with 3 subplots
        """
        # Convert to numpy and limit duration
        samples = int(duration_sec * self.sample_rate)
        
        input_np = input_audio.cpu().numpy().flatten()[:samples]
        pred_np = predicted_audio.cpu().numpy().flatten()[:samples]
        target_np = target_audio.cpu().numpy().flatten()[:samples]
        
        time = np.arange(len(input_np)) / self.sample_rate
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Input (Clean)', 'Predicted Output', 'Target Output'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=time, y=input_np, mode='lines', name='Input', line=dict(color='gray')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=time, y=pred_np, mode='lines', name='Predicted', line=dict(color='blue')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=time, y=target_np, mode='lines', name='Target', line=dict(color='green')),
            row=3, col=1
        )
        
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        fig.update_yaxes(title_text="Amplitude", row=2, col=1)
        fig.update_yaxes(title_text="Amplitude", row=3, col=1)
        
        fig.update_layout(
            title='Waveform Comparison',
            template='plotly_white',
            height=800,
            showlegend=False
        )
        
        return fig
    
    def plot_spectrogram_comparison(
        self,
        predicted_audio: torch.Tensor,
        target_audio: torch.Tensor,
        n_fft: int = 2048
    ) -> go.Figure:
        """
        Plot spectrogram comparison.
        
        Args:
            predicted_audio: Predicted waveform
            target_audio: Target waveform
            n_fft: FFT size
        
        Returns:
            Plotly figure with 2 spectrograms side-by-side
        """
        def compute_spectrogram(audio):
            spec = torch.stft(
                audio.flatten(),
                n_fft=n_fft,
                hop_length=n_fft // 4,
                window=torch.hann_window(n_fft),
                return_complex=True
            )
            mag = torch.abs(spec).cpu().numpy()
            mag_db = 20 * np.log10(mag + 1e-8)
            return mag_db
        
        pred_spec = compute_spectrogram(predicted_audio)
        target_spec = compute_spectrogram(target_audio)
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Predicted Spectrogram', 'Target Spectrogram')
        )
        
        # Time and frequency axes
        time_axis = np.arange(pred_spec.shape[1]) * (n_fft // 4) / self.sample_rate
        freq_axis = np.linspace(0, self.sample_rate / 2, pred_spec.shape[0])
        
        fig.add_trace(
            go.Heatmap(
                z=pred_spec,
                x=time_axis,
                y=freq_axis,
                colorscale='Viridis',
                name='Predicted',
                showscale=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Heatmap(
                z=target_spec,
                x=time_axis,
                y=freq_axis,
                colorscale='Viridis',
                name='Target',
                colorbar=dict(title="dB")
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_xaxes(title_text="Time (s)", row=1, col=2)
        fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=2)
        
        fig.update_layout(
            title='Spectrogram Comparison',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def plot_frequency_response(
        self,
        predicted_audio: torch.Tensor,
        target_audio: torch.Tensor
    ) -> go.Figure:
        """
        Plot frequency response comparison.
        
        Args:
            predicted_audio: Predicted waveform
            target_audio: Target waveform
        
        Returns:
            Plotly figure
        """
        # Compute FFT
        pred_fft = torch.fft.rfft(predicted_audio.flatten()).cpu().numpy()
        target_fft = torch.fft.rfft(target_audio.flatten()).cpu().numpy()
        
        # Magnitude in dB
        pred_mag_db = 20 * np.log10(np.abs(pred_fft) + 1e-8)
        target_mag_db = 20 * np.log10(np.abs(target_fft) + 1e-8)
        
        # Frequency axis
        freqs = np.fft.rfftfreq(len(predicted_audio.flatten()), 1/self.sample_rate)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=freqs,
            y=target_mag_db,
            mode='lines',
            name='Target',
            line=dict(color='green', width=2),
            opacity=0.7
        ))
        
        fig.add_trace(go.Scatter(
            x=freqs,
            y=pred_mag_db,
            mode='lines',
            name='Predicted',
            line=dict(color='blue', width=2, dash='dash'),
            opacity=0.7
        ))
        
        fig.update_layout(
            title='Frequency Response Comparison',
            xaxis_title='Frequency (Hz)',
            yaxis_title='Magnitude (dB)',
            xaxis_type='log',
            template='plotly_white',
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def create_full_report(
        self,
        output_path: Path,
        training_history: Optional[List[Dict]] = None,
        sample_audio: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ):
        """
        Create comprehensive HTML report with all visualizations.
        
        Args:
            output_path: Path to save HTML file
            training_history: Optional training history for loss curves
            sample_audio: Optional tuple of (input, predicted, target) audio for waveform plots
        """
        from plotly.subplots import make_subplots
        import plotly.io as pio
        
        # Create individual figures
        figures = []
        
        # 1. Metrics overview
        figures.append(self.plot_metrics_overview())
        
        # 2. Training history (if available)
        if training_history:
            figures.append(self.plot_training_history(training_history))
        
        # 3. Sample waveform/spectrogram (if available)
        if sample_audio:
            input_audio, pred_audio, target_audio = sample_audio
            figures.append(self.plot_waveform_comparison(input_audio, pred_audio, target_audio))
            figures.append(self.plot_spectrogram_comparison(pred_audio, target_audio))
            figures.append(self.plot_frequency_response(pred_audio, target_audio))
        
        # 4. Metric distributions (for key metrics)
        key_metrics = ['esr', 'spectral_convergence', 'phase_response_error_rad']
        for metric in key_metrics:
            if metric in self.results['metrics']:
                figures.append(self.plot_metrics_distribution(metric))
        
        # Combine into HTML
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Model Evaluation Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        .summary {{ background: white; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .plot {{ background: white; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        pre {{ background: #f8f8f8; padding: 15px; border-radius: 4px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>🎸 Guitar Effect Model Evaluation Report</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <pre>{self.results['summary']}</pre>
    </div>
    
    {''.join([f'<div class="plot">{fig.to_html(full_html=False, include_plotlyjs=False)}</div>' for fig in figures])}
    
    <footer style="text-align: center; margin-top: 40px; color: #666;">
        <p>Generated with Forger NFX Evaluation System</p>
    </footer>
</body>
</html>
"""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"✅ Interactive HTML report saved to: {output_path}")
        print(f"   Open in browser to explore results interactively!")
