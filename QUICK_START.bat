@echo off
echo 🧠 NeuroVoice 2025 - Quick Start
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "neurovoice_env" (
    echo 📦 Creating virtual environment...
    python -m venv neurovoice_env
)

echo 🔄 Activating environment...
call neurovoice_env\Scripts\activate

echo 📥 Installing dependencies...
pip install torch torchaudio streamlit numpy pandas matplotlib seaborn scikit-learn librosa soundfile pyogg

echo ✅ Setup complete!
echo.
echo 🚀 Starting NeuroVoice 2025 UI...
echo 📱 Opening browser at: http://localhost:8501
echo 🔄 Press Ctrl+C to stop the server
echo ========================================

streamlit run auto_load_ui.py --server.headless false --browser.gatherUsageStats false

pause
