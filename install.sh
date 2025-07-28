#!/bin/bash

# Madame Leota Installation Script for Raspberry Pi
# This script installs all dependencies and sets up the environment

set -e  # Exit on any error

echo "🔮 Installing Madame Leota on Raspberry Pi..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    print_warning "This doesn't appear to be a Raspberry Pi. Continuing anyway..."
fi

# Update system
print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
print_status "Installing system dependencies..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    portaudio19-dev \
    python3-pyaudio \
    espeak \
    espeak-data \
    libespeak1 \
    libespeak-dev \
    festival \
    festvox-kallpc16k \
    alsa-utils \
    pulseaudio \
    libportaudio2 \
    libportaudiocpp0 \
    ffmpeg \
    libasound2-dev \
    libsdl2-dev \
    libsdl2-mixer-2.0-0 \
    python3-opencv \
    libopencv-dev

# Create virtual environment
print_status "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
print_status "Installing Python packages..."
pip install -r requirements.txt

# Create directories
print_status "Creating directories..."
mkdir -p assets/faces
mkdir -p cache/audio
mkdir -p logs

# Copy environment file
if [ ! -f .env ]; then
    print_status "Creating .env file from template..."
    cp .env.example .env
    print_warning "Please edit .env file and add your OpenAI API key!"
fi

# Set up audio
print_status "Configuring audio..."
sudo usermod -a -G audio $USER

# Create systemd service (optional)
read -p "Do you want to create a systemd service for auto-start? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Creating systemd service..."
    
    cat > madame-leota.service << EOF
[Unit]
Description=Madame Leota Interactive Fortune Teller
After=network.target sound.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PWD
Environment=PATH=$PWD/venv/bin
ExecStart=$PWD/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    sudo mv madame-leota.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable madame-leota.service
    
    print_status "Service created. Use 'sudo systemctl start madame-leota' to start."
fi

# Test installation
print_status "Testing installation..."
python -c "
import sys
required_modules = ['cv2', 'pygame', 'pyaudio', 'openai', 'speech_recognition', 'pyttsx3']
missing = []
for module in required_modules:
    try:
        __import__(module)
        print(f'✓ {module}')
    except ImportError:
        missing.append(module)
        print(f'✗ {module}')

if missing:
    print(f'Missing modules: {missing}')
    sys.exit(1)
else:
    print('All modules installed successfully!')
"

print_status "Installation complete!"
echo
echo "📋 Next steps:"
echo "1. Edit .env file and add your OpenAI API key"
echo "2. (Optional) Add face images to assets/faces/ directory"
echo "3. Connect your projector and audio devices"
echo "4. Run: python main.py"
echo
print_warning "Make sure to reboot or log out/in for audio group changes to take effect!"
echo
echo "🎭 Enjoy your Madame Leota experience!" 