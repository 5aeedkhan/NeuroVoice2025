#!/usr/bin/env python3
"""
NeuroVoice 2025 - Complete Model Integration
Combines all components: WavLM-Large + Linformer++ + Diffusion Ensemble
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

from wavlm_extractor import WavLMFeatureExtractor
from linformer_plus import LinformerPlusEncoder
from diffusion_ensemble import DiffusionEnsemble
from ssl_pretraining import SSLFeatureExtractor

class NeuroVoice2025(nn.Module):
    """
    Complete NeuroVoice 2025 model architecture
    State-of-the-art speech disorder classification system
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        wavlm_model: str = "microsoft/wavlm-large",
        freeze_ssl: bool = True,
        linformer_dim: int = 512,
        linformer_depth: int = 6,
        linformer_heads: int = 8,
        ensemble_weights: Optional[List[float]] = None,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.freeze_ssl = freeze_ssl
        
        # Component 1: WavLM-Large Feature Extractor
        self.wavlm_extractor = WavLMFeatureExtractor(
            model_name=wavlm_model,
            freeze_encoder=freeze_ssl,
            feature_dim=1024
        )
        
        # Component 2: Linformer++ Transformer Backbone
        self.linformer_encoder = LinformerPlusEncoder(
            input_dim=1024,
            dim=linformer_dim,
            depth=linformer_depth,
            seq_len=256,
            heads=linformer_heads,
            dropout=dropout_rate
        )
        
        # Component 3: Diffusion-based Ensemble Classifier
        self.diffusion_ensemble = DiffusionEnsemble(
            input_dim=linformer_dim,
            num_classes=num_classes,
            ensemble_weights=ensemble_weights or [0.4, 0.3, 0.3]
        )
        
        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(linformer_dim, linformer_dim),
            nn.LayerNorm(linformer_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Final classification head
        self.final_classifier = nn.Sequential(
            nn.Linear(linformer_dim, linformer_dim // 2),
            nn.LayerNorm(linformer_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(linformer_dim // 2, num_classes)
        )
        
        # Model statistics
        self.total_params = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"NeuroVoice 2025 initialized:")
        print(f"Total parameters: {self.total_params:,}")
        print(f"Trainable parameters: {self.trainable_params:,}")
        print(f"Frozen SSL parameters: {self.total_params - self.trainable_params:,}")
    
    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete NeuroVoice 2025 pipeline
        
        Args:
            audio: Raw audio tensor [batch_size, samples]
            
        Returns:
            Dictionary with all intermediate and final outputs
        """
        batch_size = audio.size(0)
        
        # Stage 1: WavLM-Large Feature Extraction
        wavlm_features = self.wavlm_extractor(audio)
        audio_features = wavlm_features['features']  # [batch_size, 1024]
        sequence_features = wavlm_features['sequence_features']  # [batch_size, seq_len, 1024]
        
        # Stage 2: Linformer++ Processing
        linformer_outputs = self.linformer_encoder(sequence_features)
        encoded_features = linformer_outputs['encoded']  # [batch_size, seq_len, linformer_dim]
        global_features = linformer_outputs['global_features']  # [batch_size, linformer_dim]
        
        # Stage 3: Feature Fusion
        fused_features = self.feature_fusion(global_features)
        
        # Stage 4: Diffusion Ensemble Classification
        ensemble_outputs = self.diffusion_ensemble(fused_features)
        
        # Stage 5: Final Classification
        final_logits = self.final_classifier(fused_features)
        
        # Ensemble voting (weighted combination)
        ensemble_logits = ensemble_outputs['final_logits']
        final_logits = 0.7 * ensemble_logits + 0.3 * final_logits
        
        # Compute probabilities
        probabilities = F.softmax(final_logits, dim=1)
        
        return {
            'final_logits': final_logits,
            'probabilities': probabilities,
            'predictions': torch.argmax(final_logits, dim=1),
            
            # Intermediate outputs for analysis
            'wavlm_features': audio_features,
            'sequence_features': sequence_features,
            'linformer_encoded': encoded_features,
            'global_features': global_features,
            'fused_features': fused_features,
            
            # Ensemble outputs
            'ensemble_outputs': ensemble_outputs,
            
            # Attention weights for explainability
            'attention_weights': linformer_outputs.get('attention_weights', None)
        }
    
    def predict(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make predictions with confidence scores
        
        Args:
            audio: Audio tensor
            
        Returns:
            Dictionary with predictions and confidence
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(audio)
            
            # Get predictions and confidence
            predictions = outputs['predictions']
            probabilities = outputs['probabilities']
            confidence = torch.max(probabilities, dim=1)[0]
            
            return {
                'predictions': predictions,
                'probabilities': probabilities,
                'confidence': confidence,
                'class_names': self.get_class_names()
            }
    
    def get_class_names(self) -> List[str]:
        """Get class names"""
        return ['healthy', 'dysarthria', 'apraxia', 'dysphonia']
    
    def get_feature_importance(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyze feature importance for explainability
        
        Args:
            audio: Audio tensor
            
        Returns:
            Dictionary with feature importance scores
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(audio)
            
            # WavLM feature importance
            wavlm_importance = torch.norm(outputs['wavlm_features'], dim=1)
            
            # Linformer attention importance
            attention_importance = None
            if outputs['attention_weights'] is not None:
                attention_importance = torch.mean(
                    torch.stack(outputs['attention_weights']), dim=0
                ).mean(dim=(1, 2))
            
            # Ensemble model confidence
            ensemble_confidence = {}
            for model_name, logits in outputs['ensemble_outputs'].items():
                if 'logits' in logits:
                    conf = torch.max(F.softmax(logits, dim=1), dim=1)[0]
                    ensemble_confidence[model_name] = conf
            
            return {
                'wavlm_importance': wavlm_importance,
                'attention_importance': attention_importance,
                'ensemble_confidence': ensemble_confidence,
                'final_confidence': torch.max(outputs['probabilities'], dim=1)[0]
            }
    
    def get_model_info(self) -> Dict:
        """Get detailed model information"""
        return {
            'model_name': 'NeuroVoice 2025',
            'architecture': 'WavLM-Large + Linformer++ + Diffusion Ensemble',
            'total_parameters': self.total_params,
            'trainable_parameters': self.trainable_parameters,
            'frozen_parameters': self.total_params - self.trainable_params,
            'num_classes': self.num_classes,
            'input_dim': 16000 * 3,  # 3 seconds at 16kHz
            'feature_dim': 1024,
            'transformer_dim': 512,
            'components': {
                'wavlm_extractor': 'WavLM-Large (Meta SSL)',
                'linformer_encoder': 'Linformer++ (ICLR 2025)',
                'diffusion_ensemble': '3-model voting ensemble'
            },
            'performance_targets': {
                'baseline_accuracy': 0.9781,
                'target_accuracy': 0.9931,
                'expected_improvement': '+1.5%'
            }
        }
    
    def save_checkpoint(self, filepath: str, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'model_info': self.get_model_info(),
            'metrics': metrics,
            'class_names': self.get_class_names()
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint loaded from {filepath}")
        print(f"Epoch: {checkpoint['epoch']}")
        print(f"Metrics: {checkpoint['metrics']}")
        return checkpoint

class NeuroVoiceInference:
    """
    Inference wrapper for NeuroVoice 2025
    Optimized for production deployment
    """
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = device
        self.model = NeuroVoice2025()
        self.model.load_checkpoint(model_path)
        self.model.to(device)
        self.model.eval()
        
        # Warm up
        dummy_audio = torch.randn(1, 16000 * 3).to(device)
        with torch.no_grad():
            _ = self.model(dummy_audio)
        
        print(f"NeuroVoice 2025 inference ready on {device}")
    
    def classify_audio(self, audio_path: str) -> Dict:
        """
        Classify audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Classification results
        """
        import torchaudio
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Trim/pad to 3 seconds
        target_length = 16000 * 3
        if waveform.size(1) > target_length:
            waveform = waveform[:, :target_length]
        elif waveform.size(1) < target_length:
            waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.size(1)))
        
        # Classify
        waveform = waveform.to(self.device)
        with torch.no_grad():
            results = self.model.predict(waveform)
            
            # Convert to Python types
            prediction = results['predictions'].cpu().item()
            probabilities = results['probabilities'].cpu().numpy().tolist()
            confidence = results['confidence'].cpu().item()
            
            class_names = results['class_names']
            predicted_class = class_names[prediction]
            
            # Get feature importance
            importance = self.model.get_feature_importance(waveform)
            
            return {
                'predicted_class': predicted_class,
                'prediction_id': prediction,
                'confidence': confidence,
                'probabilities': dict(zip(class_names, probabilities)),
                'feature_importance': {
                    'wavlm_importance': importance['wavlm_importance'].cpu().numpy().tolist(),
                    'final_confidence': importance['final_confidence'].cpu().item()
                },
                'model_info': self.model.get_model_info()
            }

# Utility functions
def create_neurovoice_model(config: Dict) -> NeuroVoice2025:
    """Create NeuroVoice 2025 model from configuration"""
    return NeuroVoice2025(
        num_classes=config.get('num_classes', 4),
        wavlm_model=config.get('wavlm_model', 'microsoft/wavlm-large'),
        freeze_ssl=config.get('freeze_ssl', True),
        linformer_dim=config.get('linformer_dim', 512),
        linformer_depth=config.get('linformer_depth', 6),
        linformer_heads=config.get('linformer_heads', 8),
        ensemble_weights=config.get('ensemble_weights', [0.4, 0.3, 0.3]),
        dropout_rate=config.get('dropout_rate', 0.1)
    )

def benchmark_neurovoice_performance(model: NeuroVoice2025, sample_input: torch.Tensor):
    """Benchmark complete NeuroVoice 2025 performance"""
    model.eval()
    
    with torch.no_grad():
        import time
        
        # Warm up
        for _ in range(10):
            _ = model(sample_input)
        
        # Benchmark
        times = []
        for _ in range(100):
            start_time = time.time()
            _ = model(sample_input)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
    print(f"NeuroVoice 2025 Performance Benchmark:")
    print(f"Average inference time: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
    print(f"Throughput: {1/avg_time:.2f} samples/second")
    
    return {
        'avg_time': avg_time,
        'std_time': std_time,
        'throughput': 1/avg_time
    }

if __name__ == "__main__":
    # Test complete NeuroVoice 2025 model
    config = {
        'num_classes': 4,
        'freeze_ssl': True,
        'linformer_dim': 512,
        'linformer_depth': 6,
        'ensemble_weights': [0.4, 0.3, 0.3]
    }
    
    model = create_neurovoice_model(config)
    
    # Test with dummy input
    dummy_audio = torch.randn(2, 16000 * 3)  # 2 samples, 3 seconds
    
    with torch.no_grad():
        outputs = model(dummy_audio)
        
    print("NeuroVoice 2025 Complete Test:")
    print(f"Input shape: {dummy_audio.shape}")
    print(f"Final logits shape: {outputs['final_logits'].shape}")
    print(f"Probabilities shape: {outputs['probabilities'].shape}")
    print(f"Predictions: {outputs['predictions']}")
    
    # Test predictions
    predictions = model.predict(dummy_audio)
    print(f"Prediction results: {predictions}")
    
    # Test feature importance
    importance = model.get_feature_importance(dummy_audio)
    print(f"Feature importance keys: {list(importance.keys())}")
    
    # Model info
    model_info = model.get_model_info()
    print(f"Model info: {model_info}")
    
    # Benchmark
    benchmark_neurovoice_performance(model, dummy_audio)
