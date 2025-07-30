#!/usr/bin/env python3
"""
Madame Leota Test Runner
Simple test version for monitor testing
"""

import asyncio
import signal
import sys
import logging
from pathlib import Path

# Import test config instead of regular config
from config_test import *
from src.audio_manager import AudioManager
from src.chatgpt_client import ChatGPTClient
from src.face_animator import FaceAnimator
from src.speech_processor import SpeechProcessor
from src.display_manager import DisplayManager

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{LOGS_DIR}/madame_leota_test.log'),
        logging.StreamHandler()
    ]
)

class MadameLeotaTest:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.running = False
        
        # Initialize components
        self.audio_manager = None
        self.chatgpt_client = None
        self.face_animator = None
        self.speech_processor = None
        self.display_manager = None
        
    async def initialize(self):
        """Initialize all components"""
        try:
            print("\n🔮 MADAME LEOTA - TESTING MODE")
            print("================================")
            self.logger.info("Initializing Madame Leota - TESTING MODE...")
            
            # Create necessary directories
            Path(FACE_ASSETS_DIR).mkdir(parents=True, exist_ok=True)
            Path(AUDIO_CACHE_DIR).mkdir(parents=True, exist_ok=True)
            Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
            
            # Initialize components
            print("Initializing audio...")
            self.audio_manager = AudioManager()
            
            print("Initializing ChatGPT client...")
            self.chatgpt_client = ChatGPTClient()
            
            print("Initializing display...")
            self.display_manager = DisplayManager()
            
            print("Initializing face animator...")
            self.face_animator = FaceAnimator(self.display_manager)
            
            print("Initializing speech processor...")
            self.speech_processor = SpeechProcessor(self.audio_manager)
            
            # Test connections
            print("Testing OpenAI connection...")
            await self.chatgpt_client.test_connection()
            
            print("Testing audio...")
            self.audio_manager.test_audio()
            
            print("\n✅ TESTING MODE READY!")
            print("- Windowed display (1280x720)")
            print("- Enhanced debug logging") 
            print("- Press ESC or close window to quit")
            print("- Press F to toggle fullscreen")
            print("- Speak to Madame Leota!")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}")
            print(f"❌ Initialization failed: {e}")
            return False
    
    async def start_conversation(self):
        """Start the main conversation loop"""
        self.running = True
        
        # Initial greeting
        await self.speak_response("Welcome, mortal... I am Madame Leota. Speak to me and I shall reveal your destiny...")
        
        try:
            while self.running:
                # Handle display events (including quit)
                if not self.display_manager.handle_events():
                    self.logger.info("Display closed by user")
                    break
                
                # Listen for user input
                user_input = await self.listen_for_input()
                
                if user_input:
                    print(f"🎤 User said: {user_input}")
                    
                    # Get AI response
                    response = await self.chatgpt_client.get_response(user_input)
                    
                    if response:
                        print(f"🔮 Leota responds: {response}")
                        await self.speak_response(response)
                    else:
                        await self.speak_response("The spirits are silent... perhaps try again, dear seeker.")
                
                await asyncio.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        except Exception as e:
            self.logger.error(f"Error in conversation loop: {e}")
            print(f"❌ Error: {e}")
        finally:
            await self.shutdown()
    
    async def listen_for_input(self):
        """Listen for user speech input"""
        try:
            self.face_animator.start_idle_animation()
            text = await self.speech_processor.listen_for_speech()
            self.face_animator.stop_idle_animation()
            return text
        except Exception as e:
            self.logger.error(f"Error listening for input: {e}")
            return None
    
    async def speak_response(self, text):
        """Speak response with facial animation"""
        try:
            audio_data, phonemes = await self.speech_processor.text_to_speech_with_phonemes(text)
            
            animation_task = asyncio.create_task(
                self.face_animator.animate_speaking(phonemes)
            )
            audio_task = asyncio.create_task(
                self.audio_manager.play_audio(audio_data)
            )
            
            await asyncio.gather(animation_task, audio_task)
            self.face_animator.start_idle_animation()
            
        except Exception as e:
            self.logger.error(f"Error speaking response: {e}")
            print(f"❌ Speech error: {e}")
    
    async def shutdown(self):
        """Cleanup and shutdown"""
        print("\n🔮 Shutting down Madame Leota...")
        self.running = False
        
        if self.face_animator:
            self.face_animator.cleanup()
        if self.audio_manager:
            self.audio_manager.cleanup()
        if self.display_manager:
            self.display_manager.cleanup()

async def main():
    app = MadameLeotaTest()
    
    if await app.initialize():
        await app.start_conversation()
    else:
        print("❌ Failed to initialize. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 