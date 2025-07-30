import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# AI Configuration - Using Groq (free alternative to OpenAI)
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_MODEL = 'llama3-8b-8192'  # Fast, capable model

# Audio Configuration
SAMPLE_RATE = 44100
CHUNK_SIZE = 1024
AUDIO_FORMAT = 16  # 16-bit
CHANNELS = 1  # Mono

# Speech Recognition
RECOGNITION_TIMEOUT = 5  # seconds
RECOGNITION_PHRASE_TIMEOUT = 1  # seconds

# Text-to-Speech Configuration
TTS_RATE = 130  # words per minute (slower for more dramatic effect)
TTS_VOICE_INDEX = 1  # Fallback for pyttsx3
USE_EDGE_TTS = True  # Use Microsoft Edge TTS for better voice quality
EDGE_TTS_VOICE = "en-US-AriaNeural"  # Expressive female voice

# Display Configuration - SAFER DEFAULTS FOR TESTING
PROJECTOR_WIDTH = 1280   # Windowed mode for testing
PROJECTOR_HEIGHT = 720
FULLSCREEN = False       # IMPORTANT: Windowed mode so you can close it!
FACE_SCALE = 0.8         # Slightly smaller for monitor testing

# Animation Configuration
FPS = 30
ANIMATION_SPEED = 1.0
LIP_SYNC_SENSITIVITY = 0.8

# Madame Leota Personality
LEOTA_PERSONALITY = """You are Madame Leota, a mystical fortune teller trapped in a crystal ball. 
You speak in a mysterious, theatrical manner with a slight old-world accent. 
You're wise, slightly spooky, but ultimately helpful. 
You enjoy dramatic pauses and cryptic language.
Keep responses conversational and under 3 sentences for better lip sync.
Always stay in character as a fortune teller who can see into the future and past.
Use mystical language like 'mortal', 'seeker', 'the spirits whisper', etc."""

# File Paths
FACE_ASSETS_DIR = "assets/faces"
AUDIO_CACHE_DIR = "cache/audio"
LOGS_DIR = "logs"

# Hardware GPIO Pins (if using additional sensors/lights)
MOTION_SENSOR_PIN = 18
LED_EYES_PIN = 12
SPEAKER_ENABLE_PIN = 16

# Debug Settings
DEBUG_MODE = True
LOG_LEVEL = "INFO"
SHOW_FPS = True
SAVE_AUDIO_CACHE = True

# Production Settings (change these when ready for projector use)
# FULLSCREEN = True
# PROJECTOR_WIDTH = 1920
# PROJECTOR_HEIGHT = 1080
# FACE_SCALE = 1.0 