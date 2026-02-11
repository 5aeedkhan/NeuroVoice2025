#!/usr/bin/env python3
"""
Multi-Dataset Training Pipeline
DiCOVA2 + UA-Speech + TORGO for cross-dataset generalization
"""

import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import pandas as pd
import librosa
import os
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import soundfile as sf

class SpeechDisorderDataset(data.Dataset):
    """
    Unified dataset for multiple speech disorder datasets
    """
    
    def __init__(
        self,
        dataset_paths: Dict[str, str],
        sample_rate: int = 16000,
        max_length: float = 3.0,
        normalize: bool = True,
        augment: bool = False
    ):
        self.dataset_paths = dataset_paths
        self.sample_rate = sample_rate
        self.max_length = int(max_length * sample_rate)
        self.normalize = normalize
        self.augment = augment
        
        # Load and combine all datasets
        self.samples = []
        self.labels = []
        self.dataset_sources = []
        
        self._load_all_datasets()
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)
        
        print(f"Loaded {len(self.samples)} samples from {len(dataset_paths)} datasets")
        print(f"Classes: {self.label_encoder.classes_}")
    
    def _load_all_datasets(self):
        """Load and combine all datasets"""
        for dataset_name, dataset_path in self.dataset_paths.items():
            print(f"Loading {dataset_name} from {dataset_path}")
            
            if dataset_name.lower() == 'dicova2':
                self._load_dicova2(dataset_path, dataset_name)
            elif dataset_name.lower() == 'ua_speech':
                self._load_ua_speech(dataset_path, dataset_name)
            elif dataset_name.lower() == 'toronto':
                self._load_toronto(dataset_path, dataset_name)
            else:
                print(f"Unknown dataset: {dataset_name}")
    
    def _load_dicova2(self, dataset_path: str, source_name: str):
        """Load DiCOVA2 dataset"""
        # DiCOVA2 structure: audio_files/ and metadata.csv
        audio_dir = Path(dataset_path) / "audio_files"
        metadata_path = Path(dataset_path) / "metadata.csv"
        
        if metadata_path.exists():
            metadata = pd.read_csv(metadata_path)
            
            for _, row in metadata.iterrows():
                audio_path = audio_dir / row['filename']
                if audio_path.exists():
                    self.samples.append(str(audio_path))
                    # DiCOVA2 labels: healthy, covid_severe, covid_moderate, covid_mild
                    # Map to our 4 classes
                    label = self._map_dicova2_label(row['health_status'])
                    self.labels.append(label)
                    self.dataset_sources.append(source_name)
        else:
            # Fallback: load all audio files and assume healthy
            for audio_file in audio_dir.glob("*.wav"):
                self.samples.append(str(audio_file))
                self.labels.append("healthy")
                self.dataset_sources.append(source_name)
    
    def _load_ua_speech(self, dataset_path: str, source_name: str):
        """Load UA-Speech dataset"""
        # UA-Speech structure: dysarthria/ and control/
        dysarthria_dir = Path(dataset_path) / "dysarthria"
        control_dir = Path(dataset_path) / "control"
        
        # Load dysarthria samples
        if dysarthria_dir.exists():
            for audio_file in dysarthria_dir.rglob("*.wav"):
                self.samples.append(str(audio_file))
                self.labels.append("dysarthria")
                self.dataset_sources.append(source_name)
        
        # Load control samples (healthy)
        if control_dir.exists():
            for audio_file in control_dir.rglob("*.wav"):
                self.samples.append(str(audio_file))
                self.labels.append("healthy")
                self.dataset_sources.append(source_name)
    
    def _load_toronto(self, dataset_path: str, source_name: str):
        """Load TORGO dataset"""
        # TORGO structure: various folders with dysarthric speech
        toronto_dir = Path(dataset_path)
        
        for audio_file in toronto_dir.rglob("*.wav"):
            self.samples.append(str(audio_file))
            # TORGO contains dysarthric speech
            self.labels.append("dysarthria")
            self.dataset_sources.append(source_name)
    
    def _map_dicova2_label(self, health_status: str) -> str:
        """Map DiCOVA2 health status to our 4 classes"""
        mapping = {
            'healthy': 'healthy',
            'covid_severe': 'dysphonia',  # Voice quality issues
            'covid_moderate': 'apraxia',   # Speech planning issues
            'covid_mild': 'dysarthria'    # Motor speech issues
        }
        return mapping.get(health_status.lower(), 'healthy')
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get sample with audio and label"""
        audio_path = self.samples[idx]
        label = self.encoded_labels[idx]
        source = self.dataset_sources[idx]
        
        # Load audio
        waveform, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Preprocess audio
        waveform = self._preprocess_audio(waveform)
        
        # Data augmentation
        if self.augment and np.random.random() > 0.5:
            waveform = self._augment_audio(waveform)
        
        return {
            'audio': torch.FloatTensor(waveform),
            'label': torch.LongTensor([label]),
            'source': source,
            'audio_path': audio_path
        }
    
    def _preprocess_audio(self, waveform: np.ndarray) -> np.ndarray:
        """Preprocess audio to fixed length"""
        # Trim or pad to fixed length
        if len(waveform) > self.max_length:
            waveform = waveform[:self.max_length]
        elif len(waveform) < self.max_length:
            waveform = np.pad(waveform, (0, self.max_length - len(waveform)), mode='constant')
        
        # Normalize
        if self.normalize:
            waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
        
        return waveform
    
    def _augment_audio(self, waveform: np.ndarray) -> np.ndarray:
        """Apply data augmentation"""
        # Random augmentation
        aug_type = np.random.choice(['noise', 'pitch', 'speed', 'volume'])
        
        if aug_type == 'noise':
            # Add noise
            noise = np.random.normal(0, 0.01, len(waveform))
            waveform = waveform + noise
        
        elif aug_type == 'pitch':
            # Pitch shifting (simplified)
            pitch_shift = np.random.uniform(-0.1, 0.1)
            # In practice, use librosa.effects.pitch_shift
        
        elif aug_type == 'speed':
            # Speed perturbation
            speed_factor = np.random.uniform(0.9, 1.1)
            # In practice, use librosa.effects.time_stretch
        
        elif aug_type == 'volume':
            # Volume scaling
            volume_factor = np.random.uniform(0.8, 1.2)
            waveform = waveform * volume_factor
        
        return waveform

class MultiDatasetTrainer:
    """
    Multi-dataset training pipeline for NeuroVoice 2025
    """
    
    def __init__(
        self,
        model: nn.Module,
        dataset_config: Dict,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device
        
        # Create datasets
        self.train_dataset = SpeechDisorderDataset(
            dataset_config['train'],
            augment=True
        )
        self.val_dataset = SpeechDisorderDataset(
            dataset_config['val'],
            augment=False
        )
        self.test_dataset = SpeechDisorderDataset(
            dataset_config['test'],
            augment=False
        )
        
        # Create data loaders
        self.train_loader = data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = data.DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        self.test_loader = data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=1e-6
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.dataset_accuracies = {}
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.train_loader:
            audio = batch['audio'].to(self.device)
            labels = batch['label'].squeeze().to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(audio)
            
            if isinstance(outputs, dict):
                logits = outputs['final_logits']
            else:
                logits = outputs
            
            # Compute loss
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """Validate model performance"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        dataset_correct = {}
        dataset_total = {}
        
        with torch.no_grad():
            for batch in self.val_loader:
                audio = batch['audio'].to(self.device)
                labels = batch['label'].squeeze().to(self.device)
                sources = batch['source']
                
                # Forward pass
                outputs = self.model(audio)
                
                if isinstance(outputs, dict):
                    logits = outputs['final_logits']
                else:
                    logits = outputs
                
                # Compute loss
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                # Compute accuracy
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                # Per-dataset accuracy
                for i, source in enumerate(sources):
                    if source not in dataset_correct:
                        dataset_correct[source] = 0
                        dataset_total[source] = 0
                    
                    dataset_correct[source] += (predictions[i] == labels[i]).item()
                    dataset_total[source] += 1
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        # Compute per-dataset accuracies
        dataset_accuracies = {}
        for source in dataset_correct:
            dataset_accuracies[source] = dataset_correct[source] / dataset_total[source]
        
        return avg_loss, accuracy, dataset_accuracies
    
    def train(self) -> Dict[str, List]:
        """Complete training pipeline"""
        print("Starting multi-dataset training...")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")
        
        best_accuracy = 0.0
        patience = 10
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_accuracy, dataset_acc = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            self.dataset_accuracies[epoch] = dataset_acc
            
            # Print progress
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")
            print("Per-dataset accuracies:")
            for source, acc in dataset_acc.items():
                print(f"  {source}: {acc:.4f}")
            
            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_accuracy': best_accuracy,
                    'label_encoder': self.train_dataset.label_encoder
                }, 'best_model.pth')
                print(f"New best accuracy: {best_accuracy:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'dataset_accuracies': self.dataset_accuracies
        }
    
    def test(self) -> Dict[str, float]:
        """Test model on test set"""
        print("Testing model...")
        
        # Load best model
        checkpoint = torch.load('best_model.pth', map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        correct = 0
        total = 0
        dataset_correct = {}
        dataset_total = {}
        
        with torch.no_grad():
            for batch in self.test_loader:
                audio = batch['audio'].to(self.device)
                labels = batch['label'].squeeze().to(self.device)
                sources = batch['source']
                
                outputs = self.model(audio)
                
                if isinstance(outputs, dict):
                    logits = outputs['final_logits']
                else:
                    logits = outputs
                
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                # Per-dataset accuracy
                for i, source in enumerate(sources):
                    if source not in dataset_correct:
                        dataset_correct[source] = 0
                        dataset_total[source] = 0
                    
                    dataset_correct[source] += (predictions[i] == labels[i]).item()
                    dataset_total[source] += 1
        
        overall_accuracy = correct / total
        dataset_accuracies = {}
        
        for source in dataset_correct:
            dataset_accuracies[source] = dataset_correct[source] / dataset_total[source]
        
        print(f"Test Accuracy: {overall_accuracy:.4f}")
        print("Per-dataset test accuracies:")
        for source, acc in dataset_accuracies.items():
            print(f"  {source}: {acc:.4f}")
        
        return {
            'overall_accuracy': overall_accuracy,
            'dataset_accuracies': dataset_accuracies
        }

# Utility functions
def create_dataset_config(dataset_root: str) -> Dict[str, Dict[str, str]]:
    """Create dataset configuration"""
    return {
        'train': {
            'dicova2': f"{dataset_root}/dicova2/train",
            'ua_speech': f"{dataset_root}/ua_speech/train",
            'toronto': f"{dataset_root}/toronto/train"
        },
        'val': {
            'dicova2': f"{dataset_root}/dicova2/val",
            'ua_speech': f"{dataset_root}/ua_speech/val",
            'toronto': f"{dataset_root}/toronto/val"
        },
        'test': {
            'dicova2': f"{dataset_root}/dicova2/test",
            'ua_speech': f"{dataset_root}/ua_speech/test",
            'toronto': f"{dataset_root}/toronto/test"
        }
    }

if __name__ == "__main__":
    # Test multi-dataset pipeline
    dataset_config = {
        'train': {
            'dicova2': 'data/dicova2/train',
            'ua_speech': 'data/ua_speech/train',
            'toronto': 'data/toronto/train'
        },
        'val': {
            'dicova2': 'data/dicova2/val',
            'ua_speech': 'data/ua_speech/val',
            'toronto': 'data/toronto/val'
        },
        'test': {
            'dicova2': 'data/dicova2/test',
            'ua_speech': 'data/ua_speech/test',
            'toronto': 'data/toronto/test'
        }
    }
    
    # Create dummy model for testing
    from wavlm_extractor import WavLMClassifier
    
    model = WavLMClassifier(num_classes=4)
    
    # Test dataset creation
    try:
        dataset = SpeechDisorderDataset(dataset_config['train'])
        print(f"Dataset created successfully with {len(dataset)} samples")
    except Exception as e:
        print(f"Dataset creation failed: {e}")
        print("Please ensure dataset paths are correct")
