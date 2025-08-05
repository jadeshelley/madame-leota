"""
Simple Fake Mouth Animator
Draws a basic mouth shape that moves in real-time based on TTS audio
Perfect for testing audio-to-mouth synchronization
"""

import cv2
import numpy as np
import logging
import math
from pathlib import Path

class SimpleFakeMouth:
    """Simple fake mouth that moves with TTS audio"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.frame_counter = 0
        self.last_audio_intensity = 0.0
        
        # Mouth parameters
        self.mouth_center = (400, 300)  # Center of the screen
        self.base_width = 120
        self.base_height = 80
        
        # Animation parameters
        self.smoothing_factor = 0.3
        self.breathing_rate = 0.1
        self.breathing_amplitude = 0.02
        
        print("🎭 SIMPLE FAKE: Simple fake mouth initialized")
        self.logger.info("Simple fake mouth initialized")
    
    def generate_mouth_frame(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Generate a frame with animated fake mouth"""
        try:
            print(f"🎭 SIMPLE FAKE: Generating frame {self.frame_counter}, audio chunk: {len(audio_chunk)} samples")
            
            # Create blank frame
            frame = np.zeros((600, 800, 3), dtype=np.uint8)
            
            # Analyze audio for mouth movement
            audio_intensity = self._analyze_audio_intensity(audio_chunk)
            
            # Calculate mouth dimensions based on audio
            mouth_width, mouth_height = self._calculate_mouth_size(audio_intensity)
            
            # Draw the mouth
            self._draw_mouth(frame, mouth_width, mouth_height, audio_intensity)
            
            # Add debug info
            self._add_debug_info(frame, audio_chunk, audio_intensity, mouth_width, mouth_height)
            
            print(f"🎭 SIMPLE FAKE: Frame {self.frame_counter} complete - intensity: {audio_intensity:.3f}, mouth: {mouth_width}x{mouth_height}")
            self.frame_counter += 1
            return frame
            
        except Exception as e:
            print(f"❌ SIMPLE FAKE ERROR: {e}")
            # Return blank frame with error message
            error_frame = np.zeros((600, 800, 3), dtype=np.uint8)
            cv2.putText(error_frame, f"Simple Fake Error: {str(e)[:30]}", (50, 300), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            return error_frame
    
    def _analyze_audio_intensity(self, audio_chunk: np.ndarray) -> float:
        """Analyze audio chunk for intensity"""
        if len(audio_chunk) == 0:
            # No audio = mouth should be closed
            self.last_audio_intensity = 0.0
            print(f"🎭 SIMPLE FAKE: No audio chunk, intensity = 0.0")
            return 0.0
        
        # Calculate RMS for audio intensity
        rms = np.sqrt(np.mean(audio_chunk**2))
        print(f"🎭 SIMPLE FAKE: Audio chunk {len(audio_chunk)} samples, RMS = {rms:.4f}")
        
        # More responsive to immediate audio changes
        if rms > 0.001:  # Lower threshold for more sensitivity
            # Quick response to audio - but with much better scaling
            self.last_audio_intensity = rms * 0.2  # Much much reduced amplification
            print(f"🎭 SIMPLE FAKE: Audio detected! RMS={rms:.4f}, intensity={self.last_audio_intensity:.3f}")
        else:
            # No audio = quickly close mouth
            self.last_audio_intensity = max(0.0, self.last_audio_intensity - 0.3)  # Close even faster
            print(f"🎭 SIMPLE FAKE: Low audio, closing mouth, intensity={self.last_audio_intensity:.3f}")
        
        # No artificial variation - only respond to real audio
        intensity = self.last_audio_intensity
        final_intensity = max(0.0, min(1.0, intensity))
        print(f"🎭 SIMPLE FAKE: Final intensity = {final_intensity:.3f}")
        return final_intensity
    
    def _calculate_mouth_size(self, intensity: float) -> tuple:
        """Calculate mouth width and height based on audio intensity"""
        # Base dimensions
        base_w = self.base_width
        base_h = self.base_height
        
        # Much more dramatic response to audio - make it very obvious!
        if intensity < 0.05:
            # Closed mouth - make it tiny
            width = int(base_w * 0.1)    # Very very narrow
            height = int(base_h * 0.02)  # Very very thin
            state = "CLOSED"
        elif intensity < 0.15:
            # Slightly open - make it clearly visible
            width = int(base_w * 0.4)
            height = int(base_h * 0.25)
            state = "SLIGHTLY OPEN"
        elif intensity < 0.3:
            # Open - make it obvious
            width = int(base_w * 0.8)
            height = int(base_h * 0.6)
            state = "OPEN"
        elif intensity < 0.5:
            # Wide open - make it very obvious
            width = int(base_w * 1.1)
            height = int(base_h * 0.9)
            state = "WIDE OPEN"
        else:
            # Very wide open - make it huge!
            width = int(base_w * 1.5)
            height = int(base_h * 1.2)
            state = "VERY WIDE"
        
        # Add subtle breathing effect only when mouth is closed
        if intensity < 0.02:
            breathing = 1.0 + self.breathing_amplitude * math.sin(self.frame_counter * self.breathing_rate)
            width = int(width * breathing)
            height = int(height * breathing)
        
        print(f"🎭 SIMPLE FAKE: Intensity={intensity:.3f} -> {state} mouth: {width}x{height}")
        return width, height
    
    def _draw_mouth(self, frame: np.ndarray, width: int, height: int, intensity: float):
        """Draw the animated mouth on the frame"""
        try:
            x, y = self.mouth_center
            
            # Draw ONE simple mouth oval that changes size and color
            if intensity < 0.08:  # Lower threshold for red color
                # Closed mouth - tiny and red
                mouth_color = (0, 0, 255)  # Bright red
                cv2.ellipse(frame, (x, y), (width//2, height//2), 0, 0, 360, mouth_color, -1)
            else:
                # Open mouth - larger and green
                mouth_color = (0, 255, 0)  # Bright green
                cv2.ellipse(frame, (x, y), (width//2, height//2), 0, 0, 360, mouth_color, -1)
                
        except Exception as e:
            print(f"❌ SIMPLE FAKE: Error drawing mouth: {e}")
    
    def _add_debug_info(self, frame: np.ndarray, audio_chunk: np.ndarray, intensity: float, width: int, height: int):
        """Add debug information to the frame"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Background for text
        cv2.rectangle(frame, (5, 5), (400, 200), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (400, 200), (255, 255, 255), 2)
        
        # Add audio info
        audio_text = f"Audio: {len(audio_chunk)} samples"
        cv2.putText(frame, audio_text, (10, 30), font, 0.5, (255, 255, 255), 1)
        
        # Add intensity
        intensity_text = f"Intensity: {intensity:.3f}"
        cv2.putText(frame, intensity_text, (10, 55), font, 0.5, (255, 255, 255), 1)
        
        # Add mouth dimensions
        size_text = f"Mouth: {width}x{height}"
        cv2.putText(frame, size_text, (10, 80), font, 0.5, (255, 255, 255), 1)
        
        # Add frame counter
        frame_text = f"Frame: {self.frame_counter}"
        cv2.putText(frame, frame_text, (10, 105), font, 0.5, (255, 255, 255), 1)
        
        # Add status
        status_text = "🎭 Simple Fake Mouth"
        cv2.putText(frame, status_text, (10, 130), font, 0.5, (255, 255, 255), 1)
        
        # Add breathing indicator
        breathing = 1.0 + self.breathing_amplitude * math.sin(self.frame_counter * self.breathing_rate)
        breathing_text = f"Breathing: {breathing:.3f}"
        cv2.putText(frame, breathing_text, (10, 155), font, 0.5, (255, 255, 255), 1)
        
        # Add mouth state
        if intensity < 0.05:
            state_text = "State: Closed"
        elif intensity < 0.15:
            state_text = "State: Slightly Open"
        elif intensity < 0.3:
            state_text = "State: Open"
        elif intensity < 0.5:
            state_text = "State: Wide Open"
        else:
            state_text = "State: Very Wide"
        cv2.putText(frame, state_text, (10, 180), font, 0.5, (255, 255, 255), 1)
        
        # Add audio analysis info
        if len(audio_chunk) > 0:
            rms = np.sqrt(np.mean(audio_chunk**2))
            rms_text = f"RMS: {rms:.4f}"
        else:
            rms_text = "RMS: 0.0000 (No Audio)"
        cv2.putText(frame, rms_text, (10, 205), font, 0.5, (255, 255, 255), 1)
    
    def _bytes_to_audio_array(self, audio_data: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array"""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
            return audio_array
        except Exception as e:
            print(f"❌ SIMPLE FAKE: Error converting audio: {e}")
            return np.array([], dtype=np.float32)
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            print("🎭 SIMPLE FAKE: Cleaned up successfully")
        except Exception as e:
            print(f"❌ SIMPLE FAKE: Error during cleanup: {e}") 