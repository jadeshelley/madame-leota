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

# Volume Settings (0.0 to 1.0)
MASTER_VOLUME = 0.95      # Main volume (very loud!)
TTS_VOLUME = 0.9          # Text-to-speech volume
EFFECT_VOLUME = 0.8       # Sound effects volume

# Speech Recognition
RECOGNITION_TIMEOUT = 5  # seconds
RECOGNITION_PHRASE_TIMEOUT = 1  # seconds

# Text-to-Speech Configuration
TTS_RATE = 110  # words per minute (slower for more dramatic, mystical effect)
TTS_VOICE_INDEX = 1  # Fallback for pyttsx3
USE_EDGE_TTS = True  # Use Microsoft Edge TTS for better voice quality

# 🔮 MYSTICAL VOICE OPTIONS - Choose one:
EDGE_TTS_VOICE = "en-GB-SoniaNeural"     # British, sophisticated, mysterious
# EDGE_TTS_VOICE = "en-US-JennyNeural"   # Very expressive and dramatic
# EDGE_TTS_VOICE = "en-US-SaraNeural"    # Mature, deep, mystical
# EDGE_TTS_VOICE = "en-GB-LibbyNeural"   # British, theatrical
# EDGE_TTS_VOICE = "en-AU-NatashaNeural" # Australian, unique accent
# EDGE_TTS_VOICE = "en-US-AriaNeural"    # Original - expressive female

# Display Configuration - SAFER DEFAULTS FOR TESTING
PROJECTOR_WIDTH = 1280   # Windowed mode for testing
PROJECTOR_HEIGHT = 720
FULLSCREEN = False       # IMPORTANT: Windowed mode so you can close it!
FACE_SCALE = 0.8         # Slightly smaller for monitor testing

# Animation Configuration
FPS = 30
ANIMATION_SPEED = 1.0
LIP_SYNC_SENSITIVITY = 0.8

# Real-Time Face Manipulation (Option 4 - Most Realistic!)
USE_REALTIME_FACE_MANIPULATION = False  # Set to True when dlib is working
FACE_MANIPULATION_QUALITY = "medium"    # low, medium, high (affects Pi performance)
MOUTH_BLEND_SMOOTHNESS = 0.8            # 0.0 to 1.0 - smoothness of mouth blending

# AI Wav2Lip Integration (HIGHEST QUALITY!)
USE_WAV2LIP = True                      # Use AI-powered Wav2Lip for professional lip sync
WAV2LIP_QUALITY = "high"                # low, medium, high - model quality vs performance
WAV2LIP_BATCH_SIZE = 1                  # Frames processed per batch (1 for real-time)

# Audio-Driven Deepfake-Like Manipulation (MOST REALISTIC!)
USE_AUDIO_DRIVEN_FACE = True            # Analyzes audio waveform for realistic lip sync
AUDIO_ANALYSIS_QUALITY = "high"         # low, medium, high - affects realism vs performance
DEEPFAKE_SMOOTHING = 0.9                # 0.0 to 1.0 - temporal smoothing for natural movement
JAW_SENSITIVITY = 1.2                   # Jaw movement sensitivity to audio
LIP_SENSITIVITY = 0.8                   # Lip movement sensitivity to frequency

# Enhanced Morphing (Fallback - No dependencies needed!)
USE_ENHANCED_MORPHING = True             # Better morphing with timing effects
MORPH_STEPS = 8                         # More steps = smoother (but slower)
MORPH_TIMING = "dynamic"                # "linear" or "dynamic" easing

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