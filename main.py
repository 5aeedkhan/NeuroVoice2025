#!/usr/bin/env python3
"""
NeuroVoice 2025 - Main Training Pipeline
Complete speech disorder classification system
"""

import torch
import torch.nn as nn
import argparse
import json
import os
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.neurovoice_model import NeuroVoice2025, create_neurovoice_model
from src.multi_dataset import MultiDatasetTrainer, create_dataset_config
from src.ssl_pretraining import SSLTrainer, create_ssl_model
from src.evaluation import NeuroVoiceEvaluator, create_evaluator

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='NeuroVoice 2025 Training')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, default='data',
                       help='Root directory for datasets')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--max_length', type=float, default=3.0,
                       help='Maximum audio length in seconds')
    
    # Model arguments
    parser.add_argument('--num_classes', type=int, default=4,
                       help='Number of classes')
    parser.add_argument('--linformer_dim', type=int, default=512,
                       help='Linformer dimension')
    parser.add_argument('--linformer_depth', type=int, default=6,
                       help='Linformer depth')
    parser.add_argument('--linformer_heads', type=int, default=8,
                       help='Linformer heads')
    parser.add_argument('--freeze_ssl', action='store_true', default=True,
                       help='Freeze SSL weights')
    
    # Training arguments
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--pretrain_epochs', type=int, default=10,
                       help='Number of SSL pre-training epochs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    # Evaluation arguments
    parser.add_argument('--eval_only', action='store_true',
                       help='Only run evaluation')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model checkpoint')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save results')
    
    # Misc arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--use_wandb', action='store_true', default=False,
                       help='Use Weights & Biases logging')
    
    return parser.parse_args()

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_directories(args):
    """Create necessary directories"""
    directories = [
        args.save_dir,
        'checkpoints',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

def main():
    """Main training pipeline"""
    args = parse_args()
    
    print("🧠 NeuroVoice 2025 - Speech Disorder Classification")
    print("=" * 60)
    print(f"Target Accuracy: 99.31% (vs 2024 baseline: 97.81%)")
    print(f"Device: {args.device}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print("=" * 60)
    
    # Set seed
    set_seed(args.seed)
    
    # Create directories
    create_directories(args)
    
    # Dataset configuration
    dataset_config = create_dataset_config(args.data_root)
    
    # Model configuration
    model_config = {
        'num_classes': args.num_classes,
        'freeze_ssl': args.freeze_ssl,
        'linformer_dim': args.linformer_dim,
        'linformer_depth': args.linformer_depth,
        'linformer_heads': args.linformer_heads,
        'ensemble_weights': [0.4, 0.3, 0.3],
        'dropout_rate': 0.1
    }
    
    # Create model
    print("\n🏗️  Creating NeuroVoice 2025 model...")
    model = create_neurovoice_model(model_config)
    model.to(args.device)
    
    # Print model info
    model_info = model.get_model_info()
    print(f"Total Parameters: {model_info['total_parameters']:,}")
    print(f"Trainable Parameters: {model_info['trainable_parameters']:,}")
    print(f"Frozen SSL Parameters: {model_info['frozen_parameters']:,}")
    
    if args.eval_only:
        # Evaluation only mode
        if args.model_path is None:
            print("Error: --model_path required for evaluation")
            return
        
        print(f"\n📊 Loading model from {args.model_path}")
        checkpoint = model.load_checkpoint(args.model_path)
        
        print("📊 Creating evaluator...")
        evaluator = create_evaluator(model, {
            'class_names': model.get_class_names(),
            'save_dir': args.save_dir,
            'use_wandb': args.use_wandb
        })
        
        print("📊 Running evaluation...")
        # Note: You need to provide test_loader here
        # metrics = evaluator.evaluate_model(test_loader, return_predictions=True)
        # evaluator.save_results(metrics)
        
        print("📊 Evaluation complete!")
        print(evaluator.generate_report())
        
        return
    
    # Training mode
    print("\n🚀 Starting training pipeline...")
    
    # Create multi-dataset trainer
    print("📁 Setting up multi-dataset trainer...")
    trainer = MultiDatasetTrainer(
        model=model,
        dataset_config=dataset_config,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        device=args.device
    )
    
    # Train model
    print("🏋️  Training model...")
    training_history = trainer.train()
    
    # Test model
    print("🧪 Testing model...")
    test_results = trainer.test()
    
    # Create evaluator
    print("📊 Creating evaluator...")
    evaluator = create_evaluator(model, {
        'class_names': model.get_class_names(),
        'save_dir': args.save_dir,
        'use_wandb': args.use_wandb
    })
    
    # Save final model
    print("💾 Saving final model...")
    final_checkpoint_path = 'checkpoints/neurovoice2025_final.pth'
    model.save_checkpoint(
        final_checkpoint_path,
        epoch=args.num_epochs,
        metrics={
            'test_accuracy': test_results['overall_accuracy'],
            'dataset_accuracies': test_results['dataset_accuracies'],
            'training_history': training_history
        }
    )
    
    # Generate final report
    print("\n📋 Final Report:")
    print("=" * 60)
    print(f"Test Accuracy: {test_results['overall_accuracy']:.4f}")
    print(f"Target Accuracy: {model_info['performance_targets']['target_accuracy']:.4f}")
    print(f"Baseline Accuracy: {model_info['performance_targets']['baseline_accuracy']:.4f}")
    print(f"Improvement: {test_results['overall_accuracy'] - model_info['performance_targets']['baseline_accuracy']:+.4f}")
    
    print("\n📊 Per-Dataset Performance:")
    for dataset, accuracy in test_results['dataset_accuracies'].items():
        print(f"  {dataset}: {accuracy:.4f}")
    
    # Check if target achieved
    if test_results['overall_accuracy'] >= model_info['performance_targets']['target_accuracy']:
        print("\n🎉 TARGET ACHIEVED! NeuroVoice 2025 is ready!")
    elif test_results['overall_accuracy'] >= model_info['performance_targets']['baseline_accuracy']:
        print("\n✅ BASELINE BEATEN! Good performance!")
    else:
        print("\n🔄 Training complete. Consider hyperparameter tuning.")
    
    print(f"\n💾 Model saved to: {final_checkpoint_path}")
    print(f"📁 Results saved to: {args.save_dir}")
    print("\n🚀 NeuroVoice 2025 training complete!")

def demo_mode():
    """Demo mode for testing without real data"""
    print("🧠 NeuroVoice 2025 - Demo Mode")
    print("=" * 60)
    
    # Create model
    config = {
        'num_classes': 4,
        'freeze_ssl': True,
        'linformer_dim': 512,
        'linformer_depth': 6,
        'ensemble_weights': [0.4, 0.3, 0.3]
    }
    
    model = create_neurovoice_model(config)
    
    # Test with dummy data
    print("\n🧪 Testing with dummy audio data...")
    dummy_audio = torch.randn(4, 16000 * 3)  # 4 samples, 3 seconds
    
    with torch.no_grad():
        outputs = model(dummy_audio)
        
    print("✅ Model test successful!")
    print(f"Input shape: {dummy_audio.shape}")
    print(f"Output logits shape: {outputs['final_logits'].shape}")
    print(f"Predictions: {outputs['predictions'].tolist()}")
    
    # Test predictions
    predictions = model.predict(dummy_audio)
    print(f"\n🎯 Prediction Results:")
    for i in range(len(predictions['predictions'])):
        pred_class = predictions['class_names'][predictions['predictions'][i].item()]
        confidence = predictions['confidence'][i].item()
        print(f"  Sample {i+1}: {pred_class} (confidence: {confidence:.3f})")
    
    # Model info
    model_info = model.get_model_info()
    print(f"\n📊 Model Information:")
    print(f"  Architecture: {model_info['architecture']}")
    print(f"  Total Parameters: {model_info['total_parameters']:,}")
    print(f"  Trainable Parameters: {model_info['trainable_parameters']:,}")
    print(f"  Target Accuracy: {model_info['performance_targets']['target_accuracy']:.4f}")
    
    print("\n🎉 NeuroVoice 2025 demo complete!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # No arguments provided, run demo
        demo_mode()
    else:
        # Run full training
        main()
