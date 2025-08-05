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
            return self.last_audio_intensity
        
        # Calculate RMS for audio intensity
        rms = np.sqrt(np.mean(audio_chunk**2))
        
        # Smooth the intensity to prevent jittery movement
        self.last_audio_intensity = (self.smoothing_factor * rms + 
                                   (1 - self.smoothing_factor) * self.last_audio_intensity)
        
        # Add some natural variation
        variation = math.sin(self.frame_counter * 0.2) * 0.05
        
        intensity = self.last_audio_intensity + variation
        return max(0.0, min(1.0, intensity))
    
    def _calculate_mouth_size(self, intensity: float) -> tuple:
        """Calculate mouth width and height based on audio intensity"""
        # Base dimensions
        base_w = self.base_width
        base_h = self.base_height
        
        # Mouth opens more with higher intensity
        open_factor = intensity * 2.0  # 0 to 2x
        
        # Width changes less than height (more realistic)
        width = int(base_w * (0.8 + open_factor * 0.4))   # 80% to 160%
        height = int(base_h * (0.3 + open_factor * 1.4))  # 30% to 310%
        
        # Add breathing effect
        breathing = 1.0 + self.breathing_amplitude * math.sin(self.frame_counter * self.breathing_rate)
        width = int(width * breathing)
        height = int(height * breathing)
        
        return width, height
    
    def _draw_mouth(self, frame: np.ndarray, width: int, height: int, intensity: float):
        """Draw the animated mouth on the frame"""
        try:
            x, y = self.mouth_center
            
            # Draw mouth background (dark area)
            mouth_color = (20, 20, 20)  # Dark gray
            cv2.ellipse(frame, (x, y), (width//2, height//2), 0, 0, 360, mouth_color, -1)
            
            # Draw lip outline
            lip_color = (80, 40, 40)  # Dark red
            lip_thickness = max(2, int(4 * intensity))  # Thicker lips when more open
            cv2.ellipse(frame, (x, y), (width//2, height//2), 0, 0, 360, lip_color, lip_thickness)
            
            # Draw inner mouth detail (tongue/teeth area)
            if intensity > 0.3:  # Only show when mouth is open enough
                inner_color = (40, 20, 20)  # Darker red
                inner_width = int(width * 0.7)
                inner_height = int(height * 0.6)
                cv2.ellipse(frame, (x, y), (inner_width//2, inner_height//2), 0, 0, 360, inner_color, -1)
            
            # Add some highlights for realism
            if intensity > 0.5:
                highlight_color = (100, 50, 50)  # Lighter red
                highlight_width = int(width * 0.3)
                highlight_height = int(height * 0.2)
                highlight_y = y - height//4
                cv2.ellipse(frame, (x, highlight_y), (highlight_width//2, highlight_height//2), 0, 0, 360, highlight_color, -1)
                
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
        if intensity < 0.2:
            state_text = "State: Closed"
        elif intensity < 0.4:
            state_text = "State: Slightly Open"
        elif intensity < 0.6:
            state_text = "State: Open"
        elif intensity < 0.8:
            state_text = "State: Wide Open"
        else:
            state_text = "State: Very Wide"
        cv2.putText(frame, state_text, (10, 180), font, 0.5, (255, 255, 255), 1)
    
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