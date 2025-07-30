import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = 'gpt-3.5-turbo'

# Audio Configuration
SAMPLE_RATE = 44100
CHUNK_SIZE = 1024
AUDIO_FORMAT = 16
CHANNELS = 1

# Speech Recognition
RECOGNITION_TIMEOUT = 5
RECOGNITION_PHRASE_TIMEOUT = 1

# Text-to-Speech
TTS_RATE = 150
TTS_VOICE_INDEX = 1

# Display Configuration - TESTING MODE
PROJECTOR_WIDTH = 1280   # Smaller for testing
PROJECTOR_HEIGHT = 720   # Smaller for testing  
FULLSCREEN = False       # Windowed mode for easier testing
FACE_SCALE = 0.8         # Slightly smaller face for monitor

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
Always stay in character as a fortune teller who can see into the future and past."""

# File Paths
FACE_ASSETS_DIR = "assets/faces"
AUDIO_CACHE_DIR = "cache/audio"
LOGS_DIR = "logs"

# Hardware GPIO Pins
MOTION_SENSOR_PIN = 18
LED_EYES_PIN = 12
SPEAKER_ENABLE_PIN = 16

# Debug Settings - Enhanced for testing
DEBUG_MODE = True
LOG_LEVEL = "DEBUG"      # More verbose logging for testing
SHOW_FPS = True
SAVE_AUDIO_CACHE = True 