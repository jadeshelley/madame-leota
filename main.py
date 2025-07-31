#!/usr/bin/env python3
"""
Madame Leota Interactive Fortune Teller
Main application file
"""

print("🔄 IMPORT: Starting imports...")
import asyncio
import signal
import sys
import logging
from pathlib import Path
from typing import Optional
print("✅ IMPORT: Standard library imports done")

print("⚙️ IMPORT: Loading config...")
from config import *
print("✅ IMPORT: Config loaded")

print("🔊 IMPORT: Loading AudioManager...")
from src.audio_manager import AudioManager
print("✅ IMPORT: AudioManager loaded")

print("🧠 IMPORT: Loading ChatGPTClient...")
from src.chatgpt_client import ChatGPTClient
print("✅ IMPORT: ChatGPTClient loaded")

print("🎭 IMPORT: Loading FaceAnimator...")
from src.face_animator import FaceAnimator
print("✅ IMPORT: FaceAnimator loaded")

print("🗣️ IMPORT: Loading SpeechProcessor...")
from src.speech_processor import SpeechProcessor
print("✅ IMPORT: SpeechProcessor loaded")

print("📱 IMPORT: Loading DisplayManager...")
from src.display_manager import DisplayManager
print("✅ IMPORT: All imports complete!")

# Create necessary directories BEFORE setting up logging
print("📁 SETUP: Creating directories...")
Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
Path(FACE_ASSETS_DIR).mkdir(parents=True, exist_ok=True)
Path(AUDIO_CACHE_DIR).mkdir(parents=True, exist_ok=True)
print("✅ SETUP: Directories created")

# Setup logging (now that logs directory exists)
print("📝 SETUP: Configuring logging...")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{LOGS_DIR}/madame_leota.log'),
        logging.StreamHandler()
    ]
)
print("✅ SETUP: Logging configured")

class MadameLeotaApp:
    def __init__(self):
        print("🎯 CONSTRUCTOR: Entering MadameLeotaApp.__init__()")
        self.logger = logging.getLogger(__name__)
        print("✅ CONSTRUCTOR: Logger created")
        self.logger.info("🚀 Starting Madame Leota initialization...")
        
        # Initialize components
        try:
            print("📱 CONSTRUCTOR: About to create DisplayManager...")
            self.logger.info("📱 Initializing Display Manager...")
            self.display_manager = DisplayManager()
            print("✅ CONSTRUCTOR: DisplayManager created")
            self.logger.info("✅ Display Manager initialized")
            
            print("🔊 CONSTRUCTOR: About to create AudioManager...")
            self.logger.info("🔊 Initializing Audio Manager...")
            self.audio_manager = AudioManager()
            print("✅ CONSTRUCTOR: AudioManager created")
            self.logger.info("✅ Audio Manager initialized")
            
            print("🧠 CONSTRUCTOR: About to create ChatGPTClient...")
            self.logger.info("🧠 Initializing ChatGPT Client...")
            self.chatgpt_client = ChatGPTClient()
            print("✅ CONSTRUCTOR: ChatGPTClient created")
            self.logger.info("✅ ChatGPT Client initialized")
            
            print("🗣️ CONSTRUCTOR: About to create SpeechProcessor...")
            self.logger.info("🗣️ Initializing Speech Processor...")
            self.speech_processor = SpeechProcessor(self.audio_manager)
            print("✅ CONSTRUCTOR: SpeechProcessor created")
            self.logger.info("✅ Speech Processor initialized")
            
            print("🎭 CONSTRUCTOR: About to create FaceAnimator...")
            self.logger.info("🎭 Initializing Face Animator...")
            self.face_animator = FaceAnimator(self.display_manager)
            print("✅ CONSTRUCTOR: FaceAnimator created")
            print("📝 DEBUG: About to log Face Animator initialized...")
            self.logger.info("✅ Face Animator initialized")
            print("✅ DEBUG: Face Animator logging completed!")
            
            print("🎉 DEBUG: About to log all components initialized...")
            self.logger.info("🎉 All components initialized successfully!")
            print("✅ DEBUG: All components logging completed!")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize component: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        print("🔧 DEBUG: About to set up signal handlers...")
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        print("✅ DEBUG: Signal handlers set up!")
        
        self.running = False
        print("🏁 DEBUG: MadameLeotaApp.__init__() completed successfully!")
        
    async def initialize(self):
        """Run post-initialization setup and tests"""
        try:
            self.logger.info("🔧 Running post-initialization setup...")
            
            # Create necessary directories
            Path(FACE_ASSETS_DIR).mkdir(parents=True, exist_ok=True)
            Path(AUDIO_CACHE_DIR).mkdir(parents=True, exist_ok=True)
            Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
            
            # Test connections (components already initialized in __init__)
            self.logger.info("🔍 Testing connections...")
            await self.chatgpt_client.test_connection()
            self.audio_manager.test_audio()
            
            self.logger.info("Madame Leota initialized successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}")
            return False
    
    async def start_conversation(self):
        """Start the main conversation loop"""
        self.running = True
        self.logger.info("Madame Leota is ready to speak with visitors...")
        
        # Initial greeting
        await self.speak_response("Welcome, mortal... I am Madame Leota. What secrets do you seek from beyond the veil?")
        
        try:
            while self.running:
                # Listen for user input
                user_input = await self.listen_for_input()
                
                if user_input:
                    self.logger.info(f"User said: {user_input}")
                    
                    # Get AI response
                    response = await self.chatgpt_client.get_response(user_input)
                    
                    if response:
                        # Speak the response with animation
                        await self.speak_response(response)
                    else:
                        await self.speak_response("The spirits are silent... perhaps try again, dear seeker.")
                
                # Small delay to prevent excessive CPU usage
                await asyncio.sleep(0.1)
                
        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
        except Exception as e:
            self.logger.error(f"Error in conversation loop: {e}")
        finally:
            await self.shutdown()
    
    async def listen_for_input(self):
        """Listen for user speech input"""
        try:
            # Start idle animation
            self.face_animator.start_idle_animation()
            
            # Listen for speech
            text = await self.speech_processor.listen_for_speech()
            
            # Stop idle animation
            self.face_animator.stop_idle_animation()
            
            return text
            
        except Exception as e:
            self.logger.error(f"Error listening for input: {e}")
            return None
    
    async def speak_response(self, text):
        """Speak response with deepfake-like facial animation"""
        try:
            print(f"🗣️ SPEAK DEBUG: About to speak: {text}")
            self.logger.info(f"Leota says: {text}")
            
            print("🎵 SPEAK DEBUG: About to generate TTS audio...")
            # Generate audio and phoneme data for lip sync
            audio_data, phonemes = await self.speech_processor.text_to_speech_with_phonemes(text)
            print(f"✅ SPEAK DEBUG: TTS completed! Audio: {len(audio_data)} bytes, phonemes: {len(phonemes)}")
            self.logger.info(f"Generated audio: {len(audio_data)} bytes, phonemes: {len(phonemes)}")
            
            # Check what animation system is available - DEBUG LOGGING
            has_audio_driven = hasattr(self.face_animator, 'animate_speaking_with_audio') and self.face_animator.audio_driven_face
            
            self.logger.info(f"🔍 DEBUG - Animation System Check:")
            self.logger.info(f"  - hasattr animate_speaking_with_audio: {hasattr(self.face_animator, 'animate_speaking_with_audio')}")
            self.logger.info(f"  - audio_driven_face object: {self.face_animator.audio_driven_face}")
            self.logger.info(f"  - has_audio_driven (combined): {has_audio_driven}")
            
            if hasattr(self.face_animator, 'audio_driven_face') and self.face_animator.audio_driven_face:
                self.logger.info(f"  - base_face loaded: {self.face_animator.audio_driven_face.base_face is not None}")
                if self.face_animator.audio_driven_face.base_face is not None:
                    self.logger.info(f"  - base_face shape: {self.face_animator.audio_driven_face.base_face.shape}")
            
            # Start deepfake-like speaking animation with audio analysis
            if has_audio_driven:
                # Use audio-driven deepfake-like animation (most realistic)
                self.logger.info("🎭 Using audio-driven deepfake animation")
                animation_task = asyncio.create_task(
                    self.face_animator.animate_speaking_with_audio(audio_data, phonemes)
                )
            else:
                # Fallback to phoneme-based animation
                self.logger.info("⚠️  Using fallback phoneme-based animation")
                animation_task = asyncio.create_task(
                    self.face_animator.animate_speaking(phonemes)
                )
            
            # Play audio
            audio_task = asyncio.create_task(
                self.audio_manager.play_audio(audio_data)
            )
            
            # Wait for both to complete
            await asyncio.gather(animation_task, audio_task)
            
            # Return to idle state
            self.face_animator.start_idle_animation()
            
        except Exception as e:
            self.logger.error(f"Error speaking response: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    async def shutdown(self):
        """Cleanup and shutdown"""
        self.logger.info("Shutting down Madame Leota...")
        self.running = False
        
        if self.face_animator:
            self.face_animator.cleanup()
        if self.audio_manager:
            self.audio_manager.cleanup()
        if self.display_manager:
            self.display_manager.cleanup()
        
        self.logger.info("Farewell, until we meet again...")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\nReceived shutdown signal. Goodbye!")
    sys.exit(0)

async def main():
    print("🌟 MAIN: Starting main function...")
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    print("✅ MAIN: Signal handlers set up")
    
    # Create and run the application
    print("🏗️ MAIN: About to create MadameLeotaApp...")
    app = MadameLeotaApp()
    print("✅ MAIN: MadameLeotaApp created successfully!")
    
    print("⚙️ MAIN: About to run initialize...")
    if await app.initialize():
        print("🎬 MAIN: Starting conversation...")
        await app.start_conversation()
    else:
        print("Failed to initialize Madame Leota. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 