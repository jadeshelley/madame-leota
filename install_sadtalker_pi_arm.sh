#!/bin/bash

# SadTalker Installation Script for Raspberry Pi ARM
# This script attempts to install SadTalker with ARM-specific workarounds

set -e  # Exit on any error

echo "🎭 Installing SadTalker on Raspberry Pi ARM..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    print_error "This script is designed for Raspberry Pi ARM architecture!"
    exit 1
fi

# Check architecture
ARCH=$(uname -m)
print_status "Detected architecture: $ARCH"

if [[ "$ARCH" != "aarch64" && "$ARCH" != "armv7l" ]]; then
    print_error "Unsupported architecture: $ARCH"
    print_error "This script is for ARM64 (aarch64) or ARM32 (armv7l) only"
    exit 1
fi

# Check available memory
MEMORY_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
MEMORY_GB=$((MEMORY_KB / 1024 / 1024))
print_status "Available memory: ${MEMORY_GB}GB"

if [ $MEMORY_GB -lt 4 ]; then
    print_warning "SadTalker requires at least 4GB RAM. You have ${MEMORY_GB}GB"
    print_warning "Installation may fail or be very slow"
fi

# Update system
print_step "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
print_step "Installing system dependencies..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libhdf5-103 \
    libqtgui4 \
    libqtwebkit4 \
    libqt4-test \
    python3-pyqt5 \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    libgtk-3-dev \
    libcanberra-gtk3-module \
    libcanberra-gtk-module

# Create virtual environment
print_step "Creating Python virtual environment..."
python3 -m venv sadtalker_env
source sadtalker_env/bin/activate

# Upgrade pip
print_step "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch for ARM
print_step "Installing PyTorch for ARM architecture..."

if [[ "$ARCH" == "aarch64" ]]; then
    print_status "Installing PyTorch for ARM64..."
    
    # Try multiple PyTorch installation methods for ARM64
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    
    # Method 1: Try official ARM64 wheel
    if ! pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; then
        print_warning "Official PyTorch failed, trying alternative method..."
        
        # Method 2: Try conda-forge ARM64
        if command -v conda &> /dev/null; then
            conda install pytorch torchvision torchaudio cpuonly -c pytorch
        else
            # Method 3: Build from source (very slow)
            print_warning "Building PyTorch from source (this will take hours)..."
            pip install torch torchvision torchaudio --no-binary torch
        fi
    fi
else
    print_status "Installing PyTorch for ARM32..."
    # ARM32 is much more limited
    pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
fi

# Verify PyTorch installation
print_step "Verifying PyTorch installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print('PyTorch installation successful!')"

# Install SadTalker dependencies
print_step "Installing SadTalker dependencies..."
pip install \
    numpy \
    opencv-python \
    scipy \
    tqdm \
    yacs \
    gdown \
    imageio \
    imageio-ffmpeg \
    librosa \
    pyyaml \
    face-alignment \
    insightface \
    onnxruntime \
    gfpgan \
    basicsr \
    facexlib \
    dlib-binary \
    av \
    psutil \
    opencv-contrib-python \
    scikit-image \
    scikit-learn \
    matplotlib \
    seaborn \
    pandas \
    jupyter \
    ipykernel \
    tensorboard

# Clone SadTalker repository
print_step "Cloning SadTalker repository..."
if [ ! -d "SadTalker" ]; then
    git clone https://github.com/OpenTalker/SadTalker.git
    cd SadTalker
else
    cd SadTalker
    git pull origin main
fi

# Install SadTalker
print_step "Installing SadTalker..."
pip install -e .

# Download pre-trained models
print_step "Downloading pre-trained models..."
mkdir -p checkpoints

# Download models (this will take a while)
print_warning "Downloading models (this may take 30+ minutes)..."
python3 scripts/download_models.py

# Test SadTalker installation
print_step "Testing SadTalker installation..."
python3 -c "
import sys
sys.path.append('SadTalker')

try:
    from src.utils.preprocess import align_img
    from src.utils.audio import get_mel_from_audio
    print('✅ SadTalker core modules imported successfully!')
except ImportError as e:
    print(f'❌ SadTalker import failed: {e}')
    sys.exit(1)
"

# Create test script
print_step "Creating test script..."
cat > test_sadtalker.py << 'EOF'
#!/usr/bin/env python3
"""
SadTalker Test Script for Raspberry Pi
"""

import sys
import os
sys.path.append('SadTalker')

def test_sadtalker():
    try:
        print("🎭 Testing SadTalker installation...")
        
        # Test basic imports
        from src.utils.preprocess import align_img
        from src.utils.audio import get_mel_from_audio
        print("✅ Core modules imported")
        
        # Test PyTorch
        import torch
        print(f"✅ PyTorch {torch.__version__} working")
        
        # Test OpenCV
        import cv2
        print(f"✅ OpenCV {cv2.__version__} working")
        
        # Test numpy
        import numpy as np
        print("✅ NumPy working")
        
        print("🎉 SadTalker installation successful!")
        return True
        
    except Exception as e:
        print(f"❌ SadTalker test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_sadtalker()
    sys.exit(0 if success else 1)
EOF

chmod +x test_sadtalker.py

# Run test
print_step "Running SadTalker test..."
if python3 test_sadtalker.py; then
    print_status "🎉 SadTalker installation completed successfully!"
    print_status "You can now use SadTalker on your Raspberry Pi!"
else
    print_error "❌ SadTalker installation failed!"
    print_error "Check the error messages above for details."
    exit 1
fi

# Create usage instructions
print_step "Creating usage instructions..."
cat > SADTALKER_USAGE.md << 'EOF'
# SadTalker on Raspberry Pi - Usage Guide

## 🎭 What is SadTalker?
SadTalker is an AI-powered talking face generation system that can make any face image "speak" by syncing it with audio.

## ⚠️ Important Notes for Raspberry Pi:
- **Very slow**: Processing will take 10-30x longer than on a desktop
- **Memory intensive**: Requires at least 4GB RAM
- **CPU intensive**: Will use 100% CPU during processing
- **Not real-time**: This is for pre-recorded videos, not live streaming

## 🚀 Basic Usage:

### 1. Prepare your files:
```bash
# You need:
# - A face image (PNG/JPG)
# - An audio file (WAV/MP3)
# - Both files in the same directory
```

### 2. Run SadTalker:
```bash
cd SadTalker
python3 inference.py --driven_audio path/to/audio.wav --source_image path/to/face.png --result_dir ./results
```

### 3. Find your result:
```bash
# Check the results directory
ls -la ./results/
```

## 🔧 Performance Tips:
1. **Use smaller images** (256x256 or 512x512)
2. **Use shorter audio** (30 seconds or less)
3. **Close other applications** to free up memory
4. **Be patient** - processing can take 10-30 minutes

## 🛠️ Troubleshooting:
- If you get "out of memory" errors, try smaller images
- If it's too slow, consider using a desktop computer
- If models fail to download, check your internet connection

## 📝 Example:
```bash
# Example with your Madame Leota face
python3 inference.py \
  --driven_audio ../assets/audio/sample.wav \
  --source_image ../assets/faces/mouth_closed.png \
  --result_dir ./leota_results
```
EOF

print_status "📖 Usage guide created: SADTALKER_USAGE.md"

print_status "🎭 SadTalker installation script completed!"
print_status "Next steps:"
print_status "1. Activate environment: source sadtalker_env/bin/activate"
print_status "2. Test installation: python3 test_sadtalker.py"
print_status "3. Read usage guide: cat SADTALKER_USAGE.md"
print_status "4. Try SadTalker: cd SadTalker && python3 inference.py --help" 