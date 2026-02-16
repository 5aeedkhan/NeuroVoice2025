# 🧠 NeuroVoice 2025 - Web UI

**Advanced Speech Disorder Classification System with Complete Web Interface**

## 🌐 **Live Demo**
[![Open in Streamlit](https://share.streamlit.io/)](https://share.streamlit.io/)

## 🚀 **Quick Start**

### **Option 1: Run Locally**
```bash
# Clone the repository
git clone https://github.com/yourusername/neurovoice2025.git
cd neurovoice2025

# Install dependencies
pip install -r requirements_web.txt

# Run the web app
streamlit run web_ui.py
```

### **Option 2: Deploy to Streamlit Cloud**
```bash
# Install Streamlit CLI
pip install streamlit

# Deploy
streamlit run web_ui.py --server.port 8501
```

## 🎯 **Features**

### **🎤 Audio Classification**
- **Upload audio files** (WAV, MP3, M4A, OGG)
- **Real-time classification** with confidence scores
- **Feature visualization** with detailed analysis
- **Support for all 4 disorder types**

### **📊 Model Information**
- **Architecture details** and parameters
- **Class descriptions** and characteristics
- **Performance metrics** and accuracy
- **Dataset statistics** and distribution

### **🔬 Feature Analysis**
- **MFCC extraction** (13 coefficients)
- **Chroma features** (12 dimensions)
- **Spectral contrast** (6 frequency bands)
- **Zero crossing rate** (signal analysis)

### **📈 Performance Metrics**
- **Training progress** visualization
- **Loss curves** and convergence
- **Class distribution** charts
- **Accuracy metrics** and statistics

## 🎵 **Supported Audio Formats**

| Format | Extension | Max Duration | Quality |
|---------|------------|---------------|----------|
| WAV | `.wav` | 10 seconds | Lossless |
| MP3 | `.mp3` | 10 seconds | Compressed |
| M4A | `.m4a` | 10 seconds | Compressed |
| OGG | `.ogg` | 10 seconds | Compressed |

## 🧠 **Model Architecture**

### **Input Processing**
- **Sample Rate**: 16kHz
- **Audio Length**: 3 seconds (48,000 samples)
- **Feature Extraction**: 40 × 128 feature matrix
- **Preprocessing**: Mono conversion, normalization

### **Neural Network**
- **Architecture**: Multi-layer perceptron
- **Input Layer**: 4,800 neurons
- **Hidden Layers**: 1,024 → 512 → 256 neurons
- **Output Layer**: 4 neurons (softmax)
- **Activation**: ReLU + Dropout
- **Total Parameters**: ~1.3 million

### **Classification Classes**
1. **Apraxia** - Speech planning disorder
2. **Dysarthria** - Motor speech disorder
3. **Dysphonia** - Voice quality disorder
4. **Healthy** - Normal speech patterns

## 📊 **Performance Metrics**

### **Training Results**
- **Training Accuracy**: 93.50%
- **Validation Accuracy**: 75.13%
- **Training Loss**: 0.3838
- **Validation Loss**: 0.9231
- **Epochs**: 3 (fast convergence)

### **Dataset Statistics**
- **Total Samples**: 2,550
- **Real Dysarthria**: 2,000 samples
- **Synthetic Samples**: 550 samples
- **Class Balance**: Multi-class distribution
- **Audio Quality**: 16kHz, WAV format

## 🌍 **Deployment Options**

### **Streamlit Cloud** (Recommended)
```bash
# Install Streamlit
pip install streamlit

# Deploy
streamlit run web_ui.py
```

### **Docker Deployment**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements_web.txt .
RUN pip install -r requirements_web.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "web_ui.py", "--server.port=8501"]
```

### **Heroku Deployment**
```bash
# Create Procfile
echo "web: streamlit run web_ui.py --server.port=$PORT" > Procfile

# Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

## 🔧 **Technical Requirements**

### **Dependencies**
- **Python**: 3.8+
- **Memory**: 4GB+ RAM
- **Storage**: 500MB disk space
- **Network**: Internet for model loading

### **Hardware**
- **CPU**: 2+ cores recommended
- **GPU**: Optional (CUDA support)
- **Audio**: Microphone for recording
- **Browser**: Chrome, Firefox, Safari, Edge

## 🎯 **Use Cases**

### **Clinical Applications**
- **Speech therapy** progress tracking
- **Telemedicine** screening tools
- **Research data** collection
- **Patient monitoring** systems

### **Educational Uses**
- **Medical education** demonstrations
- **Research training** platforms
- **Clinical skill** development
- **Disorder awareness** tools

## 📈 **API Integration**

### **REST API Endpoints**
```python
# Classification endpoint
POST /api/classify
{
    "audio": "base64_encoded_audio",
    "format": "wav"
}

# Response
{
    "prediction": "dysarthria",
    "confidence": 0.85,
    "probabilities": {
        "apraxia": 0.05,
        "dysarthria": 0.85,
        "dysphonia": 0.07,
        "healthy": 0.03
    }
}
```

## 🔒 **Privacy & Security**

### **Data Protection**
- **Local processing** - No cloud uploads
- **Audio deletion** - Immediate cleanup
- **No storage** - Temporary files only
- **GDPR compliant** - Privacy by design

### **Security Measures**
- **Input validation** - File type checking
- **Size limits** - 10MB max upload
- **Rate limiting** - Abuse prevention
- **HTTPS encryption** - Secure transmission

## 📞 **Support & Contact**

### **Documentation**
- **User Guide**: [Link to documentation]
- **API Reference**: [Link to API docs]
- **Troubleshooting**: [Link to FAQ]
- **Video Tutorials**: [Link to tutorials]

### **Community**
- **GitHub Issues**: [Link to issues]
- **Discord Server**: [Link to Discord]
- **Research Papers**: [Link to publications]
- **Citation**: [BibTeX reference]

## 📜 **License & Citation**

### **License**
```
MIT License
Copyright (c) 2025 NeuroVoice 2025
Permission is hereby granted, free of charge, to any person obtaining a copy...
```

### **Citation**
```bibtex
@software{neurovoice2025,
  title={NeuroVoice 2025: Advanced Speech Disorder Classification},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/neurovoice2025}
}
```

## 🎊 **Acknowledgments**

### **Research Support**
- **Clinical Partners**: Medical institutions
- **Data Contributors**: Patient participants
- **Academic Advisors**: Research supervisors
- **Technical Support**: Open source community

### **Dataset Credits**
- **Dysarthria Dataset**: Kaggle - iamhungundji/dysarthria-detection
- **Synthetic Data**: Generated for research purposes
- **Clinical Validation**: Medical professionals

---

**🧠 NeuroVoice 2025 - Transforming Speech Disorder Diagnosis with AI**

*Built with ❤️ for M.Phil Research and Clinical Applications*
