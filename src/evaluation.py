#!/usr/bin/env python3
"""
Evaluation and Accuracy Tracking System
Real-time performance monitoring for NeuroVoice 2025
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
import json
import time
from typing import Dict, List, Optional, Tuple, Union
import wandb
from pathlib import Path

class NeuroVoiceEvaluator:
    """
    Comprehensive evaluation system for NeuroVoice 2025
    """
    
    def __init__(
        self,
        model: nn.Module,
        class_names: List[str],
        save_dir: str = "results",
        use_wandb: bool = True
    ):
        self.model = model
        self.class_names = class_names
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.use_wandb = use_wandb
        
        # Initialize metrics storage
        self.metrics_history = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': [],
            'confusion_matrices': [],
            'classification_reports': [],
            'inference_times': []
        }
        
        # Target accuracy from 2024 paper
        self.baseline_accuracy = 0.9781
        self.target_accuracy = 0.9931
        
        # Initialize wandb if enabled
        if self.use_wandb:
            wandb.init(
                project="neurovoice-2025",
                name="speech-disorder-classification",
                config={
                    "target_accuracy": self.target_accuracy,
                    "baseline_accuracy": self.baseline_accuracy,
                    "num_classes": len(class_names)
                }
            )
    
    def evaluate_model(
        self,
        test_loader,
        return_predictions: bool = False
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Comprehensive model evaluation
        
        Args:
            test_loader: Test data loader
            return_predictions: Whether to return predictions
            
        Returns:
            Dictionary with all evaluation metrics
        """
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_labels = []
        inference_times = []
        
        print("Running comprehensive evaluation...")
        
        with torch.no_grad():
            for batch in test_loader:
                audio = batch['audio']
                labels = batch['label'].squeeze()
                
                # Measure inference time
                start_time = time.time()
                
                outputs = self.model(audio)
                
                if isinstance(outputs, dict):
                    logits = outputs['final_logits']
                else:
                    logits = outputs
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Get predictions and probabilities
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        y_proba = np.array(all_probabilities)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_true, y_pred, y_proba)
        metrics['inference_times'] = inference_times
        metrics['avg_inference_time'] = np.mean(inference_times)
        
        # Store metrics
        self._store_metrics(metrics)
        
        # Generate visualizations
        self._generate_visualizations(y_true, y_pred, y_proba)
        
        # Log to wandb
        if self.use_wandb:
            self._log_to_wandb(metrics)
        
        # Check if target achieved
        self._check_target_accuracy(metrics['accuracy'])
        
        if return_predictions:
            metrics['predictions'] = y_pred
            metrics['probabilities'] = y_proba
            metrics['true_labels'] = y_true
        
        return metrics
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Calculate all evaluation metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
        
        # Per-class metrics
        metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None)
        metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None)
        metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None)
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=self.class_names, output_dict=True
        )
        
        # AUC (for multi-class)
        try:
            y_true_bin = label_binarize(y_true, classes=range(len(self.class_names)))
            metrics['auc'] = roc_auc_score(y_true_bin, y_proba, multi_class='ovr')
        except:
            metrics['auc'] = 0.0
        
        # Improvement over baseline
        metrics['improvement'] = metrics['accuracy'] - self.baseline_accuracy
        metrics['target_gap'] = self.target_accuracy - metrics['accuracy']
        
        return metrics
    
    def _store_metrics(self, metrics: Dict):
        """Store metrics in history"""
        self.metrics_history['accuracy'].append(metrics['accuracy'])
        self.metrics_history['precision'].append(metrics['precision'])
        self.metrics_history['recall'].append(metrics['recall'])
        self.metrics_history['f1'].append(metrics['f1'])
        self.metrics_history['auc'].append(metrics['auc'])
        self.metrics_history['confusion_matrices'].append(metrics['confusion_matrix'])
        self.metrics_history['classification_reports'].append(metrics['classification_report'])
        self.metrics_history['inference_times'].append(metrics['avg_inference_time'])
    
    def _generate_visualizations(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ):
        """Generate comprehensive visualizations"""
        # Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix - NeuroVoice 2025')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # ROC Curves
        plt.figure(figsize=(12, 8))
        y_true_bin = label_binarize(y_true, classes=range(len(self.class_names)))
        
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            auc = np.trapz(tpr, fpr)
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - NeuroVoice 2025')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Performance Comparison
        self._plot_performance_comparison()
        
        # Inference Time Distribution
        self._plot_inference_times()
    
    def _plot_performance_comparison(self):
        """Plot performance comparison with baseline"""
        if len(self.metrics_history['accuracy']) == 0:
            return
        
        plt.figure(figsize=(12, 6))
        
        epochs = range(1, len(self.metrics_history['accuracy']) + 1)
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.metrics_history['accuracy'], 'b-', label='NeuroVoice 2025', linewidth=2)
        plt.axhline(y=self.baseline_accuracy, color='r', linestyle='--', label='2024 Baseline')
        plt.axhline(y=self.target_accuracy, color='g', linestyle='--', label='Target')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Progression')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.metrics_history['f1'], 'g-', label='F1 Score', linewidth=2)
        plt.plot(epochs, self.metrics_history['precision'], 'r-', label='Precision', linewidth=2)
        plt.plot(epochs, self.metrics_history['recall'], 'b-', label='Recall', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Performance Metrics')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_inference_times(self):
        """Plot inference time distribution"""
        if len(self.metrics_history['inference_times']) == 0:
            return
        
        plt.figure(figsize=(10, 6))
        
        plt.hist(self.metrics_history['inference_times'], bins=20, alpha=0.7, color='skyblue')
        plt.axvline(x=np.mean(self.metrics_history['inference_times']), 
                   color='red', linestyle='--', label=f'Mean: {np.mean(self.metrics_history["inference_times"])*1000:.2f}ms')
        plt.xlabel('Inference Time (seconds)')
        plt.ylabel('Frequency')
        plt.title('Inference Time Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'inference_times.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _log_to_wandb(self, metrics: Dict):
        """Log metrics to wandb"""
        if not self.use_wandb:
            return
        
        log_dict = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'auc': metrics['auc'],
            'improvement': metrics['improvement'],
            'target_gap': metrics['target_gap'],
            'avg_inference_time': metrics['avg_inference_time']
        }
        
        # Per-class metrics
        for i, class_name in enumerate(self.class_names):
            log_dict[f'{class_name}_precision'] = metrics['precision_per_class'][i]
            log_dict[f'{class_name}_recall'] = metrics['recall_per_class'][i]
            log_dict[f'{class_name}_f1'] = metrics['f1_per_class'][i]
        
        wandb.log(log_dict)
        
        # Log confusion matrix
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=metrics['true_labels'] if 'true_labels' in metrics else None,
                preds=metrics['predictions'] if 'predictions' in metrics else None,
                class_names=self.class_names
            )
        })
    
    def _check_target_accuracy(self, accuracy: float):
        """Check if target accuracy is achieved"""
        if accuracy >= self.target_accuracy:
            print(f"🎉 TARGET ACHIEVED! Accuracy: {accuracy:.4f} (Target: {self.target_accuracy:.4f})")
            if self.use_wandb:
                wandb.alert(
                    title="Target Accuracy Achieved!",
                    text=f"NeuroVoice 2025 reached {accuracy:.4f} accuracy",
                    level=wandb.AlertLevel.SUCCESS
                )
        elif accuracy >= self.baseline_accuracy:
            print(f"✅ BASELINE BEATEN! Accuracy: {accuracy:.4f} (Baseline: {self.baseline_accuracy:.4f})")
        else:
            print(f"⚠️  Accuracy: {accuracy:.4f} (Baseline: {self.baseline_accuracy:.4f}, Target: {self.target_accuracy:.4f})")
    
    def save_results(self, metrics: Dict):
        """Save evaluation results to file"""
        results = {
            'model_performance': {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'auc': metrics['auc'],
                'improvement': metrics['improvement'],
                'target_gap': metrics['target_gap']
            },
            'per_class_performance': {},
            'inference_stats': {
                'avg_inference_time': metrics['avg_inference_time'],
                'total_samples': len(metrics['true_labels']) if 'true_labels' in metrics else 0
            },
            'targets': {
                'baseline_accuracy': self.baseline_accuracy,
                'target_accuracy': self.target_accuracy
            }
        }
        
        # Per-class performance
        for i, class_name in enumerate(self.class_names):
            results['per_class_performance'][class_name] = {
                'precision': metrics['precision_per_class'][i],
                'recall': metrics['recall_per_class'][i],
                'f1': metrics['f1_per_class'][i]
            }
        
        # Save to JSON
        with open(self.save_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save classification report
        with open(self.save_dir / 'classification_report.txt', 'w') as f:
            f.write("NeuroVoice 2025 - Classification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Overall Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Improvement over 2024: {metrics['improvement']:.4f}\n")
            f.write(f"Target Gap: {metrics['target_gap']:.4f}\n\n")
            f.write("Detailed Report:\n")
            f.write(metrics['classification_report'])
        
        print(f"Results saved to {self.save_dir}")
    
    def generate_report(self) -> str:
        """Generate comprehensive evaluation report"""
        if len(self.metrics_history['accuracy']) == 0:
            return "No evaluation data available."
        
        latest_accuracy = self.metrics_history['accuracy'][-1]
        latest_f1 = self.metrics_history['f1'][-1]
        latest_inference_time = self.metrics_history['inference_times'][-1]
        
        report = f"""
🧠 NeuroVoice 2025 - Evaluation Report
{'=' * 50}

📊 PERFORMANCE METRICS:
• Overall Accuracy: {latest_accuracy:.4f} ({latest_accuracy*100:.2f}%)
• F1 Score: {latest_f1:.4f}
• Inference Time: {latest_inference_time*1000:.2f}ms

🎯 TARGET COMPARISON:
• 2024 Baseline: {self.baseline_accuracy:.4f} ({self.baseline_accuracy*100:.2f}%)
• 2025 Target: {self.target_accuracy:.4f} ({self.target_accuracy*100:.2f}%)
• Current Performance: {latest_accuracy:.4f} ({latest_accuracy*100:.2f}%)
• Improvement: {latest_accuracy - self.baseline_accuracy:+.4f}
• Gap to Target: {self.target_accuracy - latest_accuracy:.4f}

🏆 STATUS: {'✅ TARGET ACHIEVED!' if latest_accuracy >= self.target_accuracy else '🔄 IN PROGRESS'}

📈 PERFORMANCE HISTORY:
• Best Accuracy: {max(self.metrics_history['accuracy']):.4f}
• Average Accuracy: {np.mean(self.metrics_history['accuracy']):.4f}
• Accuracy Trend: {'📈 Improving' if len(self.metrics_history['accuracy']) > 1 and self.metrics_history['accuracy'][-1] > self.metrics_history['accuracy'][-2] else '📉 Declining'}

🔬 TECHNICAL DETAILS:
• Model Architecture: WavLM-Large + Linformer++ + Diffusion Ensemble
• Training: Multi-dataset (DiCOVA2 + UA-Speech + TORGO)
• Pre-training: Self-supervised (frozen weights)
• Classes: {', '.join(self.class_names)}

📁 Files Generated:
• confusion_matrix.png - Confusion matrix visualization
• roc_curves.png - ROC curves for all classes
• performance_comparison.png - Accuracy progression
• inference_times.png - Inference time distribution
• evaluation_results.json - Detailed metrics
• classification_report.txt - Text report

{'🎉 NeuroVoice 2025 is the strongest speech disorder detection system!' if latest_accuracy >= self.target_accuracy else '🚀 Continue training to reach target accuracy!'}
"""
        
        return report

class RealTimeMonitor:
    """
    Real-time monitoring for training and inference
    """
    
    def __init__(self, update_interval: int = 10):
        self.update_interval = update_interval
        self.start_time = time.time()
        self.epoch_times = []
        self.batch_times = []
        
    def start_epoch(self):
        """Start timing an epoch"""
        self.epoch_start = time.time()
    
    def end_epoch(self, epoch: int, metrics: Dict):
        """End epoch and log metrics"""
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)
        
        if epoch % self.update_interval == 0:
            print(f"Epoch {epoch} completed in {epoch_time:.2f}s")
            print(f"Metrics: {metrics}")
    
    def start_batch(self):
        """Start timing a batch"""
        self.batch_start = time.time()
    
    def end_batch(self):
        """End batch timing"""
        batch_time = time.time() - self.batch_start
        self.batch_times.append(batch_time)
    
    def get_stats(self) -> Dict:
        """Get timing statistics"""
        return {
            'avg_epoch_time': np.mean(self.epoch_times) if self.epoch_times else 0,
            'avg_batch_time': np.mean(self.batch_times) if self.batch_times else 0,
            'total_time': time.time() - self.start_time,
            'epochs_completed': len(self.epoch_times),
            'batches_completed': len(self.batch_times)
        }

# Utility functions
def create_evaluator(model: nn.Module, config: Dict) -> NeuroVoiceEvaluator:
    """Create evaluator from configuration"""
    return NeuroVoiceEvaluator(
        model=model,
        class_names=config.get('class_names', ['healthy', 'dysarthria', 'apraxia', 'dysphonia']),
        save_dir=config.get('save_dir', 'results'),
        use_wandb=config.get('use_wandb', True)
    )

def benchmark_model_performance(model: nn.Module, sample_input: torch.Tensor, num_runs: int = 100):
    """Benchmark model performance"""
    model.eval()
    
    times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(sample_input)
            end_time = time.time()
            times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"Model Performance Benchmark ({num_runs} runs):")
    print(f"Average inference time: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
    print(f"Throughput: {1/avg_time:.2f} samples/second")
    
    return {
        'avg_time': avg_time,
        'std_time': std_time,
        'throughput': 1/avg_time
    }

if __name__ == "__main__":
    # Test evaluation system
    from diffusion_ensemble import create_diffusion_ensemble
    
    # Create dummy model
    model = create_diffusion_ensemble({'input_dim': 512, 'num_classes': 4})
    
    # Create evaluator
    evaluator = create_evaluator(model, {
        'class_names': ['healthy', 'dysarthria', 'apraxia', 'dysphonia'],
        'save_dir': 'test_results',
        'use_wandb': False
    })
    
    # Test with dummy data
    dummy_predictions = np.random.randint(0, 4, 100)
    dummy_labels = np.random.randint(0, 4, 100)
    dummy_probabilities = np.random.dirichlet([1, 1, 1, 1], 100)
    
    metrics = evaluator._calculate_metrics(dummy_labels, dummy_predictions, dummy_probabilities)
    
    print("Evaluation System Test:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Improvement: {metrics['improvement']:.4f}")
    
    # Generate report
    evaluator.metrics_history['accuracy'] = [metrics['accuracy']]
    evaluator.metrics_history['f1'] = [metrics['f1']]
    evaluator.metrics_history['inference_times'] = [0.1]
    
    report = evaluator.generate_report()
    print(report)
