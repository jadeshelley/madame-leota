"""
Real Mouth Manipulator - Actually manipulates mouth shapes in real-time
Uses OpenCV to morph and manipulate the mouth region based on audio
"""

import cv2
import numpy as np
import logging
import math
from pathlib import Path

class RealMouthManipulator:
    """Real-time mouth shape manipulator that actually modifies images"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_face = None
        self.frame_counter = 0
        self.last_audio_intensity = 0.0
        
        # Mouth manipulation parameters
        self.mouth_center = None
        self.mouth_width = 100
        self.mouth_height = 60
        
        print("🎭 REAL MOUTH: Real mouth manipulator initialized")
        self.logger.info("Real mouth manipulator initialized")
    
    def load_base_face(self, face_path: str) -> bool:
        """Load base face image for manipulation"""
        try:
            self.base_face = cv2.imread(face_path)
            if self.base_face is None:
                print(f"❌ REAL MOUTH: Failed to load base face from {face_path}")
                return False
            
            # Detect mouth region (simplified - assumes mouth is in lower center)
            height, width = self.base_face.shape[:2]
            self.mouth_center = (width // 2, int(height * 0.7))  # Mouth typically at 70% down
            
            print(f"✅ REAL MOUTH: Loaded base face {self.base_face.shape}")
            print(f"🎯 REAL MOUTH: Mouth center at {self.mouth_center}")
            return True
            
        except Exception as e:
            print(f"❌ REAL MOUTH: Error loading base face: {e}")
            return False
    
    def generate_face_for_audio_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Generate face with real-time mouth manipulation"""
        try:
            if self.base_face is None:
                # Create fallback face
                fallback_face = np.ones((512, 512, 3), dtype=np.uint8) * 128
                cv2.putText(fallback_face, "Real Mouth: No Base Face", (50, 256), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                return fallback_face
            
            # Analyze audio for mouth manipulation
            audio_intensity = self._analyze_audio_intensity(audio_chunk)
            
            # Create manipulated face
            result = self._manipulate_mouth(audio_intensity)
            
            # Add debug info
            self._add_debug_info(result, audio_chunk, audio_intensity)
            
            self.frame_counter += 1
            return result
            
        except Exception as e:
            print(f"❌ REAL MOUTH ERROR: {e}")
            return self.base_face.copy() if self.base_face is not None else np.zeros((512, 512, 3), dtype=np.uint8)
    
    def _analyze_audio_intensity(self, audio_chunk: np.ndarray) -> float:
        """Analyze audio chunk for intensity"""
        if len(audio_chunk) == 0:
            return self.last_audio_intensity
        
        # Calculate RMS for audio intensity
        rms = np.sqrt(np.mean(audio_chunk**2))
        
        # Smooth the intensity
        smoothing_factor = 0.3
        self.last_audio_intensity = (smoothing_factor * rms + 
                                   (1 - smoothing_factor) * self.last_audio_intensity)
        
        return max(0.0, min(1.0, self.last_audio_intensity))
    
    def _manipulate_mouth(self, intensity: float) -> np.ndarray:
        """Actually manipulate the mouth region based on audio intensity"""
        try:
            # Start with base face
            result = self.base_face.copy()
            
            if self.mouth_center is None:
                return result
            
            # Calculate mouth dimensions based on intensity
            base_width = self.mouth_width
            base_height = self.mouth_height
            
            # Mouth opens more with higher intensity
            open_factor = intensity * 2.0  # 0 to 2x
            mouth_width = int(base_width * (0.8 + open_factor * 0.4))  # 80% to 160%
            mouth_height = int(base_height * (0.3 + open_factor * 1.4))  # 30% to 310%
            
            # Create mouth region coordinates
            x, y = self.mouth_center
            x1 = max(0, x - mouth_width // 2)
            x2 = min(result.shape[1], x + mouth_width // 2)
            y1 = max(0, y - mouth_height // 2)
            y2 = min(result.shape[0], y + mouth_height // 2)
            
            # Extract mouth region
            mouth_region = result[y1:y2, x1:x2]
            
            if mouth_region.size == 0:
                return result
            
            # Create mouth shape mask
            mouth_mask = self._create_mouth_mask(mouth_region.shape, intensity)
            
            # Apply mouth manipulation
            manipulated_mouth = self._apply_mouth_manipulation(mouth_region, mouth_mask, intensity)
            
            # Blend manipulated mouth back into face
            result[y1:y2, x1:x2] = manipulated_mouth
            
            # Add subtle breathing effect
            result = self._add_breathing_effect(result)
            
            return result
            
        except Exception as e:
            print(f"❌ REAL MOUTH: Error in mouth manipulation: {e}")
            return self.base_face.copy()
    
    def _create_mouth_mask(self, shape: tuple, intensity: float) -> np.ndarray:
        """Create a mask for the mouth shape"""
        try:
            height, width = shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Create elliptical mouth shape
            center = (width // 2, height // 2)
            
            # Mouth shape changes with intensity
            if intensity < 0.2:
                # Closed mouth - thin line
                cv2.ellipse(mask, center, (width//4, 2), 0, 0, 360, 255, -1)
            elif intensity < 0.4:
                # Slightly open - small oval
                cv2.ellipse(mask, center, (width//3, height//4), 0, 0, 360, 255, -1)
            elif intensity < 0.6:
                # Open mouth - medium oval
                cv2.ellipse(mask, center, (width//2, height//2), 0, 0, 360, 255, -1)
            elif intensity < 0.8:
                # Wide open - large oval
                cv2.ellipse(mask, center, (width//2, height//1.5), 0, 0, 360, 255, -1)
            else:
                # Very wide open - largest oval
                cv2.ellipse(mask, center, (width//1.5, height//1.2), 0, 0, 360, 255, -1)
            
            return mask
            
        except Exception as e:
            print(f"❌ REAL MOUTH: Error creating mouth mask: {e}")
            return np.zeros(shape[:2], dtype=np.uint8)
    
    def _apply_mouth_manipulation(self, mouth_region: np.ndarray, mask: np.ndarray, intensity: float) -> np.ndarray:
        """Apply actual mouth manipulation to the region"""
        try:
            result = mouth_region.copy()
            
            # Create dark mouth interior
            mouth_color = (20, 20, 20)  # Dark gray/black for mouth interior
            
            # Apply mask to create mouth shape
            mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mouth_interior = np.full_like(mouth_region, mouth_color, dtype=np.uint8)
            
            # Blend mouth interior with original region
            alpha = 0.7  # How much of the mouth interior to show
            result = cv2.addWeighted(result, 1 - alpha, mouth_interior, alpha, 0)
            
            # Add lip outline for more realism
            lip_color = (80, 40, 40)  # Dark red for lips
            lip_thickness = max(1, int(3 * intensity))  # Thicker lips when more open
            
            # Create lip outline
            kernel = np.ones((lip_thickness, lip_thickness), np.uint8)
            lip_mask = cv2.dilate(mask, kernel, iterations=1) - mask
            lip_mask_3d = cv2.cvtColor(lip_mask, cv2.COLOR_GRAY2BGR)
            
            # Apply lip color
            lip_overlay = np.full_like(mouth_region, lip_color, dtype=np.uint8)
            result = np.where(lip_mask_3d > 0, lip_overlay, result)
            
            return result
            
        except Exception as e:
            print(f"❌ REAL MOUTH: Error applying mouth manipulation: {e}")
            return mouth_region
    
    def _add_breathing_effect(self, image: np.ndarray) -> np.ndarray:
        """Add subtle breathing animation"""
        try:
            # Subtle scale variation based on breathing
            breathing_factor = 1.0 + 0.01 * math.sin(self.frame_counter * 0.1)
            
            height, width = image.shape[:2]
            new_height = int(height * breathing_factor)
            new_width = int(width * breathing_factor)
            
            # Resize with breathing effect
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # Crop back to original size
            start_y = (new_height - height) // 2
            start_x = (new_width - width) // 2
            result = resized[start_y:start_y+height, start_x:start_x+width]
            
            return result
            
        except Exception as e:
            print(f"❌ REAL MOUTH: Error adding breathing effect: {e}")
            return image
    
    def _add_debug_info(self, image: np.ndarray, audio_chunk: np.ndarray, intensity: float):
        """Add debug information to the image"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Background for text
        cv2.rectangle(image, (5, 5), (400, 180), (0, 0, 0), -1)
        cv2.rectangle(image, (5, 5), (400, 180), (255, 255, 255), 2)
        
        # Add audio info
        audio_text = f"Audio: {len(audio_chunk)} samples"
        cv2.putText(image, audio_text, (10, 30), font, 0.5, (255, 255, 255), 1)
        
        # Add intensity
        intensity_text = f"Intensity: {intensity:.3f}"
        cv2.putText(image, intensity_text, (10, 55), font, 0.5, (255, 255, 255), 1)
        
        # Add mouth dimensions
        if self.mouth_center:
            mouth_text = f"Mouth: {self.mouth_width}x{self.mouth_height}"
            cv2.putText(image, mouth_text, (10, 80), font, 0.5, (255, 255, 255), 1)
        
        # Add frame counter
        frame_text = f"Frame: {self.frame_counter}"
        cv2.putText(image, frame_text, (10, 105), font, 0.5, (255, 255, 255), 1)
        
        # Add status
        status_text = "🎭 Real Mouth Manipulation"
        cv2.putText(image, status_text, (10, 130), font, 0.5, (255, 255, 255), 1)
        
        # Add breathing indicator
        breathing_text = f"Breathing: {1.0 + 0.01 * math.sin(self.frame_counter * 0.1):.3f}"
        cv2.putText(image, breathing_text, (10, 155), font, 0.5, (255, 255, 255), 1)
    
    def _bytes_to_audio_array(self, audio_data: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array"""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
            return audio_array
        except Exception as e:
            print(f"❌ REAL MOUTH: Error converting audio: {e}")
            return np.array([], dtype=np.float32)
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.base_face = None
            print("🎭 REAL MOUTH: Cleaned up successfully")
        except Exception as e:
            print(f"❌ REAL MOUTH: Error during cleanup: {e}") 