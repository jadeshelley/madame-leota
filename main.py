#!/usr/bin/env python3
"""
Madame Leota Interactive Fortune Teller
Main application file
"""

import asyncio
import signal
import sys
import logging
from pathlib import Path

from config import *
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
        logging.FileHandler(f'{LOGS_DIR}/madame_leota.log'),
        logging.StreamHandler()
    ]
)

class MadameLeotaApp:
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
            self.logger.info("Initializing Madame Leota...")
            
            # Create necessary directories
            Path(FACE_ASSETS_DIR).mkdir(parents=True, exist_ok=True)
            Path(AUDIO_CACHE_DIR).mkdir(parents=True, exist_ok=True)
            Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
            
            # Initialize components
            self.audio_manager = AudioManager()
            self.chatgpt_client = ChatGPTClient()
            self.display_manager = DisplayManager()
            self.face_animator = FaceAnimator(self.display_manager)
            self.speech_processor = SpeechProcessor(self.audio_manager)
            
            # Test connections
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
        """Speak response with facial animation"""
        try:
            self.logger.info(f"Leota says: {text}")
            
            # Generate audio and phoneme data for lip sync
            audio_data, phonemes = await self.speech_processor.text_to_speech_with_phonemes(text)
            
            # Start speaking animation
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
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and run the application
    app = MadameLeotaApp()
    
    if await app.initialize():
        await app.start_conversation()
    else:
        print("Failed to initialize Madame Leota. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 