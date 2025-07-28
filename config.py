import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = 'gpt-3.5-turbo'  # or 'gpt-4' for better responses

# Audio Configuration
SAMPLE_RATE = 44100
CHUNK_SIZE = 1024
AUDIO_FORMAT = 16  # 16-bit
CHANNELS = 1  # Mono

# Speech Recognition
RECOGNITION_TIMEOUT = 5  # seconds
RECOGNITION_PHRASE_TIMEOUT = 1  # seconds

# Text-to-Speech
TTS_RATE = 150  # words per minute
TTS_VOICE_INDEX = 1  # Female voice (adjust based on available voices)

# Display Configuration
PROJECTOR_WIDTH = 1920
PROJECTOR_HEIGHT = 1080
FULLSCREEN = True
FACE_SCALE = 1.0  # Adjust to fit head form

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

# Hardware GPIO Pins (if using additional sensors/lights)
MOTION_SENSOR_PIN = 18
LED_EYES_PIN = 12
SPEAKER_ENABLE_PIN = 16

# Debug Settings
DEBUG_MODE = True
LOG_LEVEL = "INFO"
SHOW_FPS = True
SAVE_AUDIO_CACHE = True 