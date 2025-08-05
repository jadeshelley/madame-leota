#!/bin/bash

echo "🎭 Installing SadTalker for Raspberry Pi..."
echo "=========================================="

# Check if we're on a Pi
if [[ $(uname -m) == "aarch64" || $(uname -m) == "armv7l" ]]; then
    echo "✅ Detected ARM architecture (Raspberry Pi)"
else
    echo "⚠️ Not on ARM architecture, but continuing..."
fi

# Method 1: Try piwheels.org (Pi-optimized packages)
echo ""
echo "🔄 Method 1: Trying piwheels.org (Pi-optimized PyTorch)..."
pip3 install torch torchvision torchaudio --index-url https://www.piwheels.org/simple

if [ $? -eq 0 ]; then
    echo "✅ PyTorch installed via piwheels.org"
    PYTORCH_AVAILABLE=true
else
    echo "❌ piwheels.org method failed"
    PYTORCH_AVAILABLE=false
fi

# Method 2: Try CPU-only PyTorch
if [ "$PYTORCH_AVAILABLE" = false ]; then
    echo ""
    echo "🔄 Method 2: Trying CPU-only PyTorch..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    if [ $? -eq 0 ]; then
        echo "✅ PyTorch installed via CPU-only method"
        PYTORCH_AVAILABLE=true
    else
        echo "❌ CPU-only method failed"
    fi
fi

# Method 3: Try conda (if available)
if [ "$PYTORCH_AVAILABLE" = false ]; then
    echo ""
    echo "🔄 Method 3: Trying conda installation..."
    if command -v conda &> /dev/null; then
        conda install pytorch torchvision torchaudio cpuonly -c pytorch
        if [ $? -eq 0 ]; then
            echo "✅ PyTorch installed via conda"
            PYTORCH_AVAILABLE=true
        else
            echo "❌ conda method failed"
        fi
    else
        echo "⚠️ conda not available, skipping"
    fi
fi

# Install SadTalker dependencies
echo ""
echo "🔄 Installing SadTalker dependencies..."

# Basic dependencies
pip3 install opencv-python numpy scipy librosa yacs gfpgan facexlib

# Try to install SadTalker
if [ "$PYTORCH_AVAILABLE" = true ]; then
    echo ""
    echo "🔄 Installing SadTalker..."
    
    # Clone SadTalker repository
    if [ ! -d "SadTalker" ]; then
        git clone https://github.com/OpenTalker/SadTalker.git
    fi
    
    cd SadTalker
    
    # Install SadTalker
    pip3 install -e .
    
    if [ $? -eq 0 ]; then
        echo "✅ SadTalker installed successfully!"
        
        # Download models
        echo ""
        echo "🔄 Downloading SadTalker models..."
        python3 download_models.py
        
        echo ""
        echo "🎉 SadTalker installation complete!"
        echo "You can now use the SadTalker animator in Madame Leota."
        
    else
        echo "❌ SadTalker installation failed"
        echo "Will use fallback animation system"
    fi
    
    cd ..
else
    echo ""
    echo "⚠️ PyTorch not available - SadTalker cannot be installed"
    echo "The system will use the fallback animation method"
fi

# Test installation
echo ""
echo "🔄 Testing installation..."
python3 -c "
try:
    import torch
    print('✅ PyTorch available')
    try:
        import sys
        sys.path.append('SadTalker')
        from src.utils.preprocess import align_img
        print('✅ SadTalker available')
    except ImportError:
        print('❌ SadTalker not available')
except ImportError:
    print('❌ PyTorch not available')
"

echo ""
echo "🎭 Installation script complete!"
echo "Check the output above to see what's available." 