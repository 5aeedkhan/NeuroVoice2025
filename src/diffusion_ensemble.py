#!/usr/bin/env python3
"""
Diffusion-based Ensemble Classifier
Most stable & highest accuracy classification system
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

class DiffusionClassifier(nn.Module):
    """
    Diffusion-based classifier using denoising diffusion process
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        num_classes: int = 4,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        
        # Noise schedule
        self.beta = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, axis=0)
        
        # Class embeddings
        self.class_embedding = nn.Embedding(num_classes, hidden_dim)
        
        # Time embeddings
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Denoising network
        self.denoise_net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through diffusion classifier
        
        Args:
            x: Input features [batch_size, input_dim]
            t: Optional timestep [batch_size]
            
        Returns:
            Dictionary with logits and denoised features
        """
        batch_size = x.size(0)
        
        # Random timestep if not provided
        if t is None:
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=x.device)
        
        # Add noise to input
        noise = torch.randn_like(x)
        alpha_t = self.alpha_cumprod[t].view(-1, 1)
        noisy_x = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise
        
        # Time embedding
        time_emb = self.time_embedding(t.float().view(-1, 1) / self.num_timesteps)
        
        # Class embedding (using predicted class)
        with torch.no_grad():
            pred_logits = self.classifier(x)
            pred_class = torch.argmax(pred_logits, dim=1)
        class_emb = self.class_embedding(pred_class)
        
        # Concatenate for denoising
        denoise_input = torch.cat([noisy_x, time_emb, class_emb], dim=1)
        
        # Denoise
        denoised = self.denoise_net(denoise_input)
        
        # Classification
        logits = self.classifier(denoised)
        
        return {
            'logits': logits,
            'denoised': denoised,
            'noisy': noisy_x,
            'timestep': t
        }

class XGBoostStyleNet(nn.Module):
    """
    Neural network inspired by XGBoost's gradient boosting approach
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        num_classes: int = 4,
        num_trees: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        
        # Multiple decision tree-like networks
        self.trees = nn.ModuleList([
            self._create_tree(input_dim, max_depth)
            for _ in range(num_trees)
        ])
        
        # Final aggregation layer
        self.aggregator = nn.Linear(num_trees * num_classes, num_classes)
    
    def _create_tree(self, input_dim: int, max_depth: int) -> nn.Module:
        """Create a tree-like network"""
        layers = []
        current_dim = input_dim
        
        for depth in range(max_depth):
            layers.extend([
                nn.Linear(current_dim, current_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(current_dim * 2, current_dim)
            ])
            current_dim = current_dim // 2 if depth > 0 else current_dim
        
        # Final output layer
        layers.append(nn.Linear(current_dim, self.num_classes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through gradient boosting style network"""
        tree_outputs = []
        
        for tree in self.trees:
            output = tree(x)
            tree_outputs.append(output)
        
        # Concatenate all tree outputs
        all_outputs = torch.cat(tree_outputs, dim=1)
        
        # Aggregate with learning rate
        aggregated = self.aggregator(all_outputs) * self.learning_rate
        
        return {
            'logits': aggregated,
            'tree_outputs': tree_outputs
        }

class ResidualEnsembleNet(nn.Module):
    """
    Residual ensemble network for stable classification
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        num_classes: int = 4,
        num_experts: int = 3,
        expert_dim: int = 256
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_experts = num_experts
        
        # Multiple expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(expert_dim, expert_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(expert_dim, num_classes)
            )
            for _ in range(num_experts)
        ])
        
        # Gating network for expert selection
        self.gate = nn.Sequential(
            nn.Linear(input_dim, expert_dim),
            nn.ReLU(),
            nn.Linear(expert_dim, num_experts),
            nn.Softmax(dim=1)
        )
        
        # Residual connection
        self.residual = nn.Linear(input_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through residual ensemble"""
        # Get expert outputs
        expert_outputs = []
        for expert in self.experts:
            output = expert(x)
            expert_outputs.append(output)
        
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch, experts, classes]
        
        # Gating weights
        gate_weights = self.gate(x)  # [batch, experts]
        gate_weights = gate_weights.unsqueeze(-1)  # [batch, experts, 1]
        
        # Weighted combination of experts
        weighted_output = torch.sum(expert_outputs * gate_weights, dim=1)
        
        # Residual connection
        residual_output = self.residual(x)
        
        # Final output
        final_output = weighted_output + residual_output
        
        return {
            'logits': final_output,
            'expert_outputs': expert_outputs,
            'gate_weights': gate_weights.squeeze(-1)
        }

class DiffusionEnsemble(nn.Module):
    """
    Complete Diffusion-based Ensemble Classifier
    Combines 3 strong models for highest accuracy
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        num_classes: int = 4,
        ensemble_weights: Optional[List[float]] = None
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Initialize three strong models
        self.diffusion_model = DiffusionClassifier(input_dim, num_classes)
        self.xgb_style_model = XGBoostStyleNet(input_dim, num_classes)
        self.residual_ensemble = ResidualEnsembleNet(input_dim, num_classes)
        
        # Ensemble weights (default: equal voting)
        if ensemble_weights is None:
            self.ensemble_weights = torch.tensor([0.4, 0.3, 0.3])
        else:
            self.ensemble_weights = torch.tensor(ensemble_weights)
        
        # Final voting layer
        self.voting_layer = nn.Linear(num_classes * 3, num_classes)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through diffusion ensemble
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Dictionary with ensemble predictions and individual model outputs
        """
        # Get predictions from all models
        diff_outputs = self.diffusion_model(x)
        xgb_outputs = self.xgb_style_model(x)
        res_outputs = self.residual_ensemble(x)
        
        # Extract logits
        diff_logits = diff_outputs['logits']
        xgb_logits = xgb_outputs['logits']
        res_logits = res_outputs['logits']
        
        # Weighted ensemble
        weighted_logits = (
            self.ensemble_weights[0] * diff_logits +
            self.ensemble_weights[1] * xgb_logits +
            self.ensemble_weights[2] * res_logits
        )
        
        # Concatenate for final voting
        all_logits = torch.cat([diff_logits, xgb_logits, res_logits], dim=1)
        final_logits = self.voting_layer(all_logits)
        
        # Ensemble prediction (majority voting style)
        ensemble_probs = F.softmax(final_logits, dim=1)
        
        return {
            'final_logits': final_logits,
            'ensemble_probs': ensemble_probs,
            'weighted_logits': weighted_logits,
            'diffusion_logits': diff_logits,
            'xgb_logits': xgb_logits,
            'residual_logits': res_logits,
            'individual_outputs': {
                'diffusion': diff_outputs,
                'xgb_style': xgb_outputs,
                'residual': res_outputs
            }
        }
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions with ensemble"""
        with torch.no_grad():
            outputs = self.forward(x)
            predictions = torch.argmax(outputs['final_logits'], dim=1)
        return predictions
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities"""
        with torch.no_grad():
            outputs = self.forward(x)
            return outputs['ensemble_probs']
    
    def get_model_confidence(self, x: torch.Tensor) -> Dict[str, float]:
        """Get confidence scores for each model"""
        with torch.no_grad():
            outputs = self.forward(x)
            
            # Calculate confidence for each model
            diff_conf = torch.max(F.softmax(outputs['diffusion_logits'], dim=1), dim=1)[0].mean().item()
            xgb_conf = torch.max(F.softmax(outputs['xgb_logits'], dim=1), dim=1)[0].mean().item()
            res_conf = torch.max(F.softmax(outputs['residual_logits'], dim=1), dim=1)[0].mean().item()
            ensemble_conf = torch.max(outputs['ensemble_probs'], dim=1)[0].mean().item()
            
            return {
                'diffusion_confidence': diff_conf,
                'xgb_style_confidence': xgb_conf,
                'residual_confidence': res_conf,
                'ensemble_confidence': ensemble_conf
            }

# Utility functions
def create_diffusion_ensemble(config: Dict) -> DiffusionEnsemble:
    """Create diffusion ensemble from configuration"""
    return DiffusionEnsemble(
        input_dim=config.get('input_dim', 512),
        num_classes=config.get('num_classes', 4),
        ensemble_weights=config.get('ensemble_weights', [0.4, 0.3, 0.3])
    )

def benchmark_ensemble_performance(model: DiffusionEnsemble, sample_input: torch.Tensor):
    """Benchmark ensemble performance"""
    model.eval()
    
    with torch.no_grad():
        import time
        start_time = time.time()
        
        for _ in range(100):
            _ = model(sample_input)
        
        avg_time = (time.time() - start_time) / 100
        
    print(f"Diffusion Ensemble average inference time: {avg_time*1000:.2f}ms")
    return avg_time

if __name__ == "__main__":
    # Test Diffusion Ensemble
    config = {
        'input_dim': 512,
        'num_classes': 4,
        'ensemble_weights': [0.4, 0.3, 0.3]
    }
    
    model = create_diffusion_ensemble(config)
    
    # Test with dummy input
    dummy_input = torch.randn(4, 512)
    
    with torch.no_grad():
        outputs = model(dummy_input)
        
    print("Diffusion Ensemble Test:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Final logits shape: {outputs['final_logits'].shape}")
    print(f"Ensemble probs shape: {outputs['ensemble_probs'].shape}")
    
    # Test predictions
    predictions = model.predict(dummy_input)
    probabilities = model.predict_proba(dummy_input)
    confidence = model.get_model_confidence(dummy_input)
    
    print(f"Predictions: {predictions}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Model confidence: {confidence}")
    
    # Benchmark
    benchmark_ensemble_performance(model, dummy_input)
