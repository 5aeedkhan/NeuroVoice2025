#!/usr/bin/env python3
"""
NeuroVoice 2025 - Main Web App with Cloud Model
Lightweight GitHub deployment with Google Drive model loading
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
import gdown
import os
from pathlib import Path
import plotly.graph_objects as go
import io
import time

# Page configuration
st.set_page_config(
    page_title="NeuroVoice 2025 - Live Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .cloud-badge {
        background: linear-gradient(45deg, #4ECDC4, #44A08D);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9ff;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

class NeuroVoiceCloudModel:
    """Cloud-based NeuroVoice model with Google Drive integration"""
    
    def __init__(self):
        self.model = None
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        self.model_path = self.model_dir / "ultra_simple_neurovoice_model.pth"
        
        # Google Drive file ID (UPDATE THIS WITH YOUR FILE ID)
        self.drive_file_id = "1O_TqHYS3SaK9OITDQ6liMqBXIQ0qzqKq"
        
        # Class names
        self.class_names = ['apraxia', 'dysarthria', 'dysphonia', 'healthy']
    
    def download_model_from_drive(self):
        """Download model from Google Drive"""
        if self.drive_file_id == "YOUR_GOOGLE_DRIVE_FILE_ID_HERE":
            return False
        
        try:
            url = f"https://drive.google.com/uc?id={self.drive_file_id}"
            
            with st.spinner("📥 Downloading model from Google Drive..."):
                gdown.download(url, str(self.model_path), quiet=False)
            
            return self.model_path.exists()
        except Exception as e:
            st.error(f"❌ Download failed: {e}")
            return False
    
    def load_model(self):
        """Load model from local file or download from cloud"""
        # Check if model exists locally
        if self.model_path.exists():
            try:
                self.model = torch.load(self.model_path, map_location='cpu')
                self.model.eval()
                return True
            except Exception as e:
                st.error(f"❌ Error loading model: {e}")
                return False
        
        # Try to download from cloud
        if self.download_model_from_drive():
            return self.load_model()
        
        return False
    
    def extract_features(self, audio, sr=16000):
        """Extract features from audio"""
        # Ensure consistent length
        max_length = 3 * sr
        if len(audio) > max_length:
            audio = audio[:max_length]
        else:
            audio = np.pad(audio, (0, max_length - len(audio)))
        
        # Simple feature extraction (flattened for model)
        return audio.astype(np.float32)
    
    def classify_audio(self, audio, sr=16000):
        """Classify audio using the loaded model"""
        if self.model is None:
            return None, None, None
        
        try:
            features = self.extract_features(audio, sr)
            
            # For the simple model, we need to flatten features
            if len(features.shape) > 1:
                features = features.flatten()
            
            # Ensure correct size (48000 for our model)
            target_size = 48000
            if len(features) < target_size:
                features = np.pad(features, (0, target_size - len(features)))
            else:
                features = features[:target_size]
            
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1)
                
                confidence_scores = probabilities.numpy()[0]
                prediction = self.class_names[predicted_class.item()]
                confidence = confidence_scores[predicted_class.item()]
            
            return prediction, confidence, confidence_scores
            
        except Exception as e:
            st.error(f"❌ Classification error: {e}")
            return None, None, None

def create_main_interface():
    """Create the main web interface"""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 NeuroVoice 2025</h1>', unsafe_allow_html=True)
    st.markdown('<div class="cloud-badge">☁️ CLOUD-POWERED LIVE DEMO</div>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #666;">Advanced Speech Disorder Classification System</h3>', unsafe_allow_html=True)
    
    # Initialize model
    model_manager = NeuroVoiceCloudModel()
    
    # Sidebar
    st.sidebar.markdown("## 🎛️ Control Panel")
    
    # Model status
    st.sidebar.markdown("### 📊 Model Status")
    
    if model_manager.load_model():
        st.sidebar.success("✅ Model loaded successfully!")
        st.sidebar.markdown(f"**File:** {model_manager.model_path.name}")
        st.sidebar.markdown(f"**Size:** {model_manager.model_path.stat().st_size / (1024*1024):.1f} MB")
        model_ready = True
    else:
        st.sidebar.error("❌ Model not available")
        st.sidebar.markdown("**Status:** Download required")
        
        # Show setup instructions
        with st.sidebar.expander("🔧 Setup Instructions"):
            st.markdown("""
            1. **Upload model to Google Drive**
            2. **Get File ID** from share link
            3. **Update** `drive_file_id` in code
            4. **Restart** the app
            """)
        
        model_ready = False
    
    # Main content
    if model_ready:
        # Audio classification section
        st.markdown("## 🎤 Audio Classification")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # File upload
            uploaded_file = st.file_uploader(
                "Upload Audio File",
                type=['wav', 'mp3', 'm4a', 'ogg'],
                help="Upload an audio file for speech disorder classification"
            )
            
            if uploaded_file is not None:
                # Display audio player
                st.audio(uploaded_file, format='audio/wav')
                
                # Process audio
                with st.spinner("🔄 Analyzing audio..."):
                    try:
                        # Read audio
                        audio_bytes = uploaded_file.read()
                        audio, sr = sf.read(io.BytesIO(audio_bytes))
                        
                        # Convert to mono
                        if len(audio.shape) > 1:
                            audio = np.mean(audio, axis=1)
                        
                        # Classify
                        prediction, confidence, probabilities = model_manager.classify_audio(audio, sr)
                        
                        if prediction is not None:
                            # Display results
                            st.markdown("### 🎯 Classification Result")
                            col_result1, col_result2 = st.columns(2)
                            
                            with col_result1:
                                st.markdown(f"#### **{prediction.upper()}**")
                                st.markdown(f"**Confidence:** {confidence:.1%}")
                            
                            with col_result2:
                                # Confidence chart
                                fig = go.Figure(data=[
                                    go.Bar(
                                        x=model_manager.class_names,
                                        y=probabilities,
                                        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
                                        text=[f'{p:.1%}' for p in probabilities],
                                        textposition='auto',
                                    )
                                ])
                                
                                fig.update_layout(
                                    title="Classification Confidence",
                                    xaxis_title="Disorder Type",
                                    yaxis_title="Confidence (%)",
                                    yaxis=dict(tickformat='.0%'),
                                    height=300
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error processing audio: {e}")
        
        with col2:
            # Model information
            st.markdown("### 📊 Model Information")
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**🧠 Architecture**")
            st.markdown("- Type: Neural Network")
            st.markdown("- Parameters: ~1.3M")
            st.markdown("- Classes: 4")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**🎯 Classes**")
            for i, cls in enumerate(model_manager.class_names):
                st.markdown(f"**{i+1}.** {cls.title()}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**📈 Performance**")
            st.markdown("- Training: 93.5%")
            st.markdown("- Validation: 75.1%")
            st.markdown("- Dataset: 2,550 samples")
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Setup instructions when model is not ready
        st.markdown("## 🔧 Model Setup Required")
        
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        st.markdown("### 📥 Model Download Required")
        st.markdown("The model file needs to be downloaded from Google Drive.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed setup steps
        col_setup1, col_setup2 = st.columns(2)
        
        with col_setup1:
            st.markdown("### 📤 Upload Steps")
            st.markdown("""
            1. **Go to Google Drive**
            2. **Upload** `ultra_simple_neurovoice_model.pth`
            3. **Right-click** → "Get link"
            4. **Copy FILE_ID** from URL
            5. **Update** the code
            """)
        
        with col_setup2:
            st.markdown("### 🔗 URL Format")
            st.markdown("""
            **Share Link:**
            ```
            https://drive.google.com/file/d/FILE_ID/view?usp=sharing
            ```
            
            **Extract FILE_ID** (between `/d/` and `/view`)
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <strong>NeuroVoice 2025</strong> - Cloud-Powered Speech Classification<br>
        ☁️ Google Drive Storage | 🧠 Deep Learning Model | 🎯 Clinical Applications<br>
        <em>M.Phil Research Project | Live Demo | GitHub Deployment</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    create_main_interface()
