#!/usr/bin/env python3
"""
WavLM-Large Audio Feature Extractor
Meta's best self-supervised audio model for speech disorder classification
"""

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from transformers import Wav2Vec2Processor, WavLMModel
from typing import Dict, List, Optional, Tuple

class WavLMFeatureExtractor(nn.Module):
    """
    WavLM-Large feature extractor for speech disorder classification
    Uses frozen self-supervised weights for superior representation learning
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/wavlm-large",
        freeze_encoder: bool = True,
        feature_dim: int = 1024,
        sample_rate: int = 16000,
        max_length: float = 3.0
    ):
        super().__init__()
        
        self.model_name = model_name
        self.feature_dim = feature_dim
        self.sample_rate = sample_rate
        self.max_length = int(max_length * sample_rate)
        self.freeze_encoder = freeze_encoder
        
        # Load WavLM model and processor
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.wavlm = WavLMModel.from_pretrained(model_name)
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.wavlm.parameters():
                param.requires_grad = False
        
        # Feature projection layer
        self.feature_projection = nn.Sequential(
            nn.Linear(self.wavlm.config.hidden_size, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Multi-scale feature extraction
        self.multi_scale_pooling = MultiScalePooling()
        
    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract WavLM features from audio
        
        Args:
            audio: Raw audio tensor [batch_size, samples]
            
        Returns:
            Dictionary containing:
            - 'features': Main features [batch_size, feature_dim]
            - 'sequence_features': Sequence features [batch_size, seq_len, feature_dim]
            - 'attention_mask': Attention mask [batch_size, seq_len]
        """
        batch_size = audio.size(0)
        
        # Process audio to correct length
        audio = self._preprocess_audio(audio)
        
        # WavLM forward pass
        with torch.set_grad_enabled(not self.freeze_encoder):
            outputs = self.wavlm(
                audio,
                output_attentions=True,
                output_hidden_states=True
            )
        
        # Extract hidden states
        hidden_states = outputs.hidden_states  # List of [batch_size, seq_len, hidden_size]
        
        # Multi-scale feature extraction
        multi_scale_features = self.multi_scale_pooling(hidden_states)
        
        # Project to target dimension
        projected_features = self.feature_projection(multi_scale_features)
        
        # Extract sequence features (last hidden state)
        sequence_features = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        sequence_features = self.feature_projection(sequence_features)
        
        # Global pooling for final features
        final_features = torch.mean(sequence_features, dim=1)  # [batch_size, feature_dim]
        
        return {
            'features': final_features,
            'sequence_features': sequence_features,
            'attention_mask': outputs.attentions[-1] if outputs.attentions else None,
            'hidden_states': hidden_states
        }
    
    def _preprocess_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Preprocess audio to correct format and length"""
        # Ensure correct sample rate (assuming input is already at 16kHz)
        if audio.size(-1) > self.max_length:
            # Truncate
            audio = audio[:, :self.max_length]
        elif audio.size(-1) < self.max_length:
            # Pad
            pad_length = self.max_length - audio.size(-1)
            audio = torch.nn.functional.pad(audio, (0, pad_length), mode='constant', value=0)
        
        return audio
    
    def extract_features_batch(self, audio_files: List[str]) -> torch.Tensor:
        """
        Extract features from a batch of audio files
        
        Args:
            audio_files: List of audio file paths
            
        Returns:
            Feature tensor [num_files, feature_dim]
        """
        features = []
        
        for file_path in audio_files:
            # Load audio
            waveform, sample_rate = torchaudio.load(file_path)
            
            # Resample if needed
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Extract features
            with torch.no_grad():
                feature_dict = self.forward(waveform)
                features.append(feature_dict['features'])
        
        return torch.stack(features, dim=0)
    
    def get_feature_importance(self, audio: torch.Tensor) -> Dict[str, float]:
        """
        Analyze feature importance for explainability
        
        Args:
            audio: Audio tensor
            
        Returns:
            Dictionary of feature importance scores
        """
        with torch.no_grad():
            outputs = self.forward(audio)
            
            # Calculate attention-based importance
            if outputs['attention_mask'] is not None:
                attention_weights = torch.mean(outputs['attention_mask'], dim=1)
                importance = torch.mean(attention_weights, dim=1).squeeze()
            else:
                # Fallback to gradient-based importance
                importance = torch.norm(outputs['features'], dim=1)
        
        return {
            f'feature_{i}': float(imp) 
            for i, imp in enumerate(importance.cpu().numpy())
        }

class MultiScalePooling(nn.Module):
    """Multi-scale pooling for better feature extraction"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """
        Perform multi-scale pooling on hidden states
        
        Args:
            hidden_states: List of hidden states from different layers
            
        Returns:
            Pooled features [batch_size, seq_len, hidden_size]
        """
        # Use last 4 layers for multi-scale features
        last_layers = hidden_states[-4:]
        
        # Average pooling across layers
        multi_scale = torch.stack(last_layers, dim=0).mean(dim=0)
        
        return multi_scale

class WavLMClassifier(nn.Module):
    """
    Complete WavLM-based classifier for speech disorders
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        dropout_rate: float = 0.2,
        use_ssl_features: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_ssl_features = use_ssl_features
        
        # WavLM feature extractor
        self.wavlm_extractor = WavLMFeatureExtractor()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for classification
        
        Args:
            audio: Audio tensor [batch_size, samples]
            
        Returns:
            Dictionary with logits and features
        """
        # Extract WavLM features
        features = self.wavlm_extractor(audio)
        
        # Classification
        logits = self.classifier(features['features'])
        
        return {
            'logits': logits,
            'features': features['features'],
            'sequence_features': features['sequence_features']
        }

# Utility functions
def load_wavlm_model(model_path: Optional[str] = None) -> WavLMFeatureExtractor:
    """Load WavLM model with optional checkpoint"""
    model = WavLMFeatureExtractor()
    
    if model_path:
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        print(f"Loaded WavLM model from {model_path}")
    
    return model

def benchmark_wavlm_performance(model: WavLMFeatureExtractor, sample_audio: torch.Tensor):
    """Benchmark WavLM performance"""
    model.eval()
    
    with torch.no_grad():
        import time
        start_time = time.time()
        
        for _ in range(100):
            _ = model(sample_audio)
        
        avg_time = (time.time() - start_time) / 100
        
    print(f"Average inference time: {avg_time*1000:.2f}ms")
    return avg_time

if __name__ == "__main__":
    # Test WavLM extractor
    model = WavLMFeatureExtractor()
    
    # Test with dummy audio
    dummy_audio = torch.randn(2, 16000 * 3)  # 2 samples, 3 seconds
    
    with torch.no_grad():
        outputs = model(dummy_audio)
        
    print("WavLM Feature Extractor Test:")
    print(f"Features shape: {outputs['features'].shape}")
    print(f"Sequence features shape: {outputs['sequence_features'].shape}")
    print(f"Feature dimension: {outputs['features'].shape[-1]}")
    
    # Benchmark
    benchmark_wavlm_performance(model, dummy_audio)
