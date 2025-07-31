"""
Audio Manager for Madame Leota
Handles audio playback and recording
"""

import asyncio
import logging
import pygame
import pyaudio
import wave
import numpy as np
from typing import Optional
import io
from config import *

class AudioManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize pygame mixer for audio playback with Pi-compatible settings
        try:
            print("🔊 AUDIO DEBUG: Attempting to initialize pygame mixer...")
            # Try to force ALSA output
            import os
            os.environ['SDL_AUDIODRIVER'] = 'alsa'
            
            pygame.mixer.pre_init(
                frequency=22050,  # Lower sample rate for Pi compatibility
                size=-16,
                channels=1,       # Mono for better Pi compatibility
                buffer=512        # Smaller buffer for Pi
            )
            pygame.mixer.init()
            print("✅ AUDIO DEBUG: pygame mixer initialized successfully")
        except Exception as e:
            print(f"❌ AUDIO DEBUG: pygame mixer init failed: {e}")
            # Try fallback settings
            try:
                pygame.mixer.init()
                print("✅ AUDIO DEBUG: pygame mixer initialized with fallback settings")
            except Exception as e2:
                print(f"❌ AUDIO DEBUG: All audio init failed: {e2}")
        
        # Set default volume to maximum
        self.set_volume(MASTER_VOLUME)
        
        # Initialize PyAudio for recording
        self.audio = pyaudio.PyAudio()
        self.recording_stream = None
        
        self.logger.info(f"Audio Manager initialized with volume: {MASTER_VOLUME}")
    
    def test_audio(self):
        """Test audio system"""
        try:
            # Test pygame mixer
            pygame.mixer.get_init()
            
            # Test PyAudio
            device_count = self.audio.get_device_count()
            self.logger.info(f"Found {device_count} audio devices")
            
            # Find default input/output devices
            default_input = self.audio.get_default_input_device_info()
            default_output = self.audio.get_default_output_device_info()
            
            self.logger.info(f"Default input: {default_input['name']}")
            self.logger.info(f"Default output: {default_output['name']}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Audio test failed: {e}")
            return False
    
    async def play_audio(self, audio_data: bytes):
        """Play audio data asynchronously"""
        try:
            if not audio_data:
                self.logger.warning("No audio data to play")
                return
            
            # Convert bytes to BytesIO for pygame
            audio_buffer = io.BytesIO(audio_data)
            
            # Load and play the sound
            sound = pygame.mixer.Sound(audio_buffer)
            channel = sound.play()
            
            # Wait for playback to complete
            while channel.get_busy():
                await asyncio.sleep(0.01)
            
            self.logger.debug("Audio playback completed")
            
        except Exception as e:
            self.logger.error(f"Audio playback error: {e}")
    
    def play_audio_sync(self, audio_data: bytes):
        """Play audio data synchronously"""
        try:
            if not audio_data:
                return
            
            audio_buffer = io.BytesIO(audio_data)
            sound = pygame.mixer.Sound(audio_buffer)
            sound.play()
            
        except Exception as e:
            self.logger.error(f"Sync audio playback error: {e}")
    
    async def play_audio_file(self, file_path: str):
        """Play an audio file asynchronously"""
        try:
            sound = pygame.mixer.Sound(file_path)
            channel = sound.play()
            
            while channel.get_busy():
                await asyncio.sleep(0.01)
                
        except Exception as e:
            self.logger.error(f"File playback error: {e}")
    
    def get_audio_level(self, audio_data: bytes) -> float:
        """Get the audio level (volume) from audio data"""
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Calculate RMS (root mean square) for volume level
            rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            
            # Normalize to 0-1 range
            normalized = min(1.0, rms / 10000.0)
            
            return normalized
            
        except Exception as e:
            self.logger.error(f"Audio level calculation error: {e}")
            return 0.0
    
    def set_volume(self, volume: float):
        """Set global volume (0.0 to 1.0)"""
        try:
            volume = max(0.0, min(1.0, volume))
            pygame.mixer.music.set_volume(volume)
            
        except Exception as e:
            self.logger.error(f"Volume setting error: {e}")
    
    def start_recording(self) -> bool:
        """Start recording audio"""
        try:
            self.recording_stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Recording start error: {e}")
            return False
    
    def stop_recording(self) -> Optional[bytes]:
        """Stop recording and return audio data"""
        try:
            if not self.recording_stream:
                return None
            
            # Read remaining data
            frames = []
            try:
                while True:
                    data = self.recording_stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    if not data:
                        break
                    frames.append(data)
            except:
                pass  # Expected when stream ends
            
            self.recording_stream.stop_stream()
            self.recording_stream.close()
            self.recording_stream = None
            
            if frames:
                return b''.join(frames)
            return None
            
        except Exception as e:
            self.logger.error(f"Recording stop error: {e}")
            return None
    
    def save_audio_to_file(self, audio_data: bytes, filename: str):
        """Save audio data to WAV file"""
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_data)
                
        except Exception as e:
            self.logger.error(f"Audio save error: {e}")
    
    def cleanup(self):
        """Cleanup audio resources"""
        try:
            if self.recording_stream:
                self.recording_stream.stop_stream()
                self.recording_stream.close()
            
            pygame.mixer.quit()
            self.audio.terminate()
            
            self.logger.info("Audio Manager cleaned up")
            
        except Exception as e:
            self.logger.error(f"Audio cleanup error: {e}") 