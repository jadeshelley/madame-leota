# 🔮 Madame Leota Interactive Fortune Teller

Transform your Raspberry Pi into an interactive Madame Leota experience! This project creates an AI-powered fortune teller that projects an animated face onto a head form, complete with lip-sync speech and ChatGPT-powered conversations.

## ✨ Features

- **AI-Powered Conversations**: Uses OpenAI's ChatGPT for intelligent, in-character responses
- **Lip-Sync Animation**: Face animation synchronized with speech
- **Projector Display**: Optimized for short-throw projectors onto head forms
- **Speech Recognition**: Voice input for natural conversations
- **Mystical Effects**: Glowing and breathing animations for atmosphere
- **Raspberry Pi Optimized**: Designed for efficient operation on Pi hardware

## 🎬 Demo

The system creates an interactive experience where visitors can speak to Madame Leota and receive mystical fortune-telling responses with realistic facial animation.

## 🛠 Hardware Requirements

### Essential
- **Raspberry Pi 4** (4GB+ RAM recommended)
- **Short-throw projector** (compatible with head form)
- **USB microphone** or Pi camera with mic
- **Speakers** or audio output device
- **Head form** or mannequin head for projection surface

### Optional
- **PIR motion sensor** (for visitor detection)
- **LED strips** (for additional lighting effects)
- **External sound card** (for better audio quality)

## 📦 Software Requirements

- **Raspberry Pi OS** (Bullseye or newer)
- **Python 3.8+**
- **OpenAI API key** (for ChatGPT integration)

## 🚀 Quick Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd madame_leota
```

### 2. Run Installation Script
```bash
chmod +x install.sh
./install.sh
```

### 3. Configure Environment
```bash
# Copy and edit environment file
cp .env.example .env
nano .env  # Add your OpenAI API key
```

### 4. Run the Application
```bash
# Activate virtual environment
source venv/bin/activate

# Start Madame Leota
python main.py
```

## ⚙️ Manual Installation

If you prefer manual installation:

### System Dependencies
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git \
    portaudio19-dev python3-pyaudio espeak espeak-data \
    libespeak1 libespeak-dev festival alsa-utils \
    pulseaudio libsdl2-dev libsdl2-mixer-2.0-0 \
    python3-opencv libopencv-dev ffmpeg
```

### Python Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 🔧 Configuration

### Environment Variables (.env)
```bash
# Required
OPENAI_API_KEY=your_api_key_here

# Optional customization
OPENAI_MODEL=gpt-3.5-turbo  # or gpt-4 for better responses
PROJECTOR_WIDTH=1920
PROJECTOR_HEIGHT=1080
FACE_SCALE=1.0              # Adjust face size for your head form
TTS_RATE=150               # Speech speed
DEBUG_MODE=True
```

### Hardware Configuration (config.py)
- **Display settings**: Adjust projector resolution and positioning
- **Audio settings**: Configure microphone and speaker parameters
- **Animation settings**: Tune frame rate and effect intensity
- **GPIO pins**: Set up optional sensors and lighting

## 🎭 Usage

### Basic Operation
1. **Start the application**: `python main.py`
2. **Calibration**: The system will calibrate the microphone
3. **Interaction**: Speak to Madame Leota when she's in idle mode
4. **Exit**: Press Ctrl+C or Escape key

### Voice Commands
- Speak naturally to Madame Leota
- She'll respond in character as a mystical fortune teller
- Ask about the future, past, or seek mystical advice

### Keyboard Controls
- **Escape**: Exit the application
- **F**: Toggle fullscreen mode

## 🖼️ Custom Face Assets

### Adding Your Own Face Images
1. Create face images (PNG format, 400x500 recommended)
2. Name them by mouth shape:
   - `mouth_closed.png`
   - `mouth_open.png`
   - `mouth_wide.png`
   - `mouth_round.png`
   - `mouth_narrow.png`
3. Place in `assets/faces/` directory

### Face Requirements
- **Format**: PNG with transparency support
- **Size**: 400x500 pixels (will be scaled automatically)
- **Background**: Transparent or black
- **Style**: Should match Madame Leota's ethereal appearance

## 🔄 Auto-Start Setup

### Systemd Service
The installation script can create a systemd service for automatic startup:

```bash
# Enable auto-start
sudo systemctl enable madame-leota.service

# Manual control
sudo systemctl start madame-leota.service
sudo systemctl stop madame-leota.service
sudo systemctl status madame-leota.service
```

### Boot Configuration
Add to `/etc/rc.local` for simpler auto-start:
```bash
cd /path/to/madame_leota && /path/to/madame_leota/venv/bin/python main.py &
```

## 🐛 Troubleshooting

### Audio Issues
```bash
# Test audio devices
arecord -l  # List recording devices
aplay -l    # List playback devices

# Test microphone
arecord -d 5 test.wav && aplay test.wav

# Fix permissions
sudo usermod -a -G audio $USER
```

### Display Issues
```bash
# Check display
xrandr  # List displays

# Test pygame
python -c "import pygame; pygame.init(); print('Pygame working')"
```

### OpenAI Connection
```bash
# Test API key
python -c "
import openai
openai.api_key = 'your_key_here'
try:
    openai.Model.list()
    print('OpenAI connection successful')
except:
    print('OpenAI connection failed')
"
```

### Performance Issues
- **Reduce face scale**: Lower `FACE_SCALE` in config
- **Lower frame rate**: Reduce `FPS` setting
- **Disable debug mode**: Set `DEBUG_MODE=False`
- **Use lighter TTS**: Adjust `TTS_RATE`

## 📁 Project Structure

```
madame_leota/
├── main.py                 # Main application entry point
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── install.sh             # Installation script
├── .env.example          # Environment template
├── README.md             # This file
├── src/                  # Source code modules
│   ├── __init__.py
│   ├── audio_manager.py      # Audio handling
│   ├── chatgpt_client.py     # OpenAI integration
│   ├── display_manager.py    # Projector/display control
│   ├── face_animator.py      # Animation and lip-sync
│   └── speech_processor.py   # Speech recognition/TTS
├── assets/               # Media assets
│   └── faces/           # Face image files
├── cache/               # Temporary files
│   └── audio/          # Audio cache
└── logs/                # Application logs
```

## 🔮 Advanced Features

### Motion Detection
Connect a PIR sensor to automatically activate when visitors approach:
```python
# Add to config.py
MOTION_SENSOR_PIN = 18
```

### LED Effects
Add LED strips for ambient lighting:
```python
# Add to config.py
LED_EYES_PIN = 12
```

### Multiple Voices
Configure different TTS voices for variety:
```python
# In speech_processor.py
voices = self.tts_engine.getProperty('voices')
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test on Raspberry Pi
5. Submit a pull request

## 📄 License

This project is open source. See LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by Disney's Haunted Mansion Madame Leota
- OpenAI for ChatGPT API
- OpenCV and pygame communities
- Raspberry Pi Foundation

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review logs in `logs/` directory
3. Open an issue on GitHub

---

*"Rap on a table, it's time to respond. Send us a message from somewhere beyond..."* 🎭✨ 