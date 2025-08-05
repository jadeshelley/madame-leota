"""
Live2D-Style Face Animator for Madame Leota
Uses 2D face manipulation techniques that work well on Raspberry Pi
"""

import cv2
import numpy as np
import logging
import os
from pathlib import Path
from typing import Optional, Tuple
import math

class Live2DAnimator:
    """Live2D-style face animator using 2D transformations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_face = None
        self.frame_counter = 0
        
        # Animation parameters
        self.mouth_open = 0.0
        self.eye_blink = 0.0
        self.head_tilt = 0.0
        self.breathing_scale = 1.0
        
        # Face regions (estimated positions)
        self.mouth_region = None
        self.left_eye_region = None
        self.right_eye_region = None
        
        print("🎭 LIVE2D: Live2D-style animator initialized")
        self.logger.info("Live2D animator initialized")
    
    def load_base_face(self, face_image_path: str) -> bool:
        """Load the base face image and detect regions"""
        try:
            self.base_face = cv2.imread(face_image_path)
            if self.base_face is not None:
                print(f"✅ LIVE2D: Loaded base face {self.base_face.shape}")
                
                # Calculate face regions (estimated positions)
                self._calculate_face_regions()
                
                return True
            else:
                print(f"❌ LIVE2D: Failed to load {face_image_path}")
                return False
        except Exception as e:
            print(f"❌ LIVE2D: Error loading face: {e}")
            return False
    
    def _calculate_face_regions(self):
        """Calculate face regions for animation"""
        if self.base_face is None:
            return
        
        height, width = self.base_face.shape[:2]
        
        # Mouth region (lower third of face)
        mouth_y = int(height * 0.7)
        mouth_height = int(height * 0.15)
        self.mouth_region = {
            'x': int(width * 0.3),
            'y': mouth_y,
            'width': int(width * 0.4),
            'height': mouth_height
        }
        
        # Eye regions (upper third of face)
        eye_y = int(height * 0.25)
        eye_height = int(height * 0.1)
        
        self.left_eye_region = {
            'x': int(width * 0.25),
            'y': eye_y,
            'width': int(width * 0.15),
            'height': eye_height
        }
        
        self.right_eye_region = {
            'x': int(width * 0.6),
            'y': eye_y,
            'width': int(width * 0.15),
            'height': eye_height
        }
        
        print(f"✅ LIVE2D: Calculated face regions")
    
    def generate_face_for_audio_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Generate animated face based on audio using Live2D-style techniques"""
        try:
            if self.base_face is None:
                return np.zeros((512, 512, 3), dtype=np.uint8)
            
            # Analyze audio for animation parameters
            audio_intensity = self._analyze_audio(audio_chunk)
            
            # Update animation parameters
            self._update_animation_params(audio_intensity)
            
            # Create animated face
            result = self._apply_animations()
            
            # Add debug info
            self._add_debug_info(result)
            
            self.frame_counter += 1
            return result
            
        except Exception as e:
            print(f"❌ LIVE2D ERROR: {e}")
            return self.base_face.copy() if self.base_face is not None else np.zeros((512, 512, 3), dtype=np.uint8)
    
    def _analyze_audio(self, audio_chunk: np.ndarray) -> float:
        """Analyze audio chunk for animation intensity"""
        if len(audio_chunk) == 0:
            return 0.0
        
        # Calculate RMS (Root Mean Square) for audio intensity
        rms = np.sqrt(np.mean(audio_chunk**2))
        
        # Add frame-based variation for natural movement
        frame_variation = math.sin(self.frame_counter * 0.3) * 0.1
        intensity = rms + frame_variation
        
        return max(0.0, min(1.0, intensity))
    
    def _update_animation_params(self, audio_intensity: float):
        """Update animation parameters based on audio"""
        # Mouth opening (responds to audio intensity)
        target_mouth = audio_intensity * 0.8
        self.mouth_open = self.mouth_open * 0.8 + target_mouth * 0.2  # Smooth interpolation
        
        # Eye blinking (independent of audio)
        blink_cycle = (self.frame_counter % 60) / 60.0
        if blink_cycle > 0.95:  # Blink every ~60 frames
            self.eye_blink = 1.0
        else:
            self.eye_blink = max(0.0, self.eye_blink - 0.1)
        
        # Head tilt (subtle movement)
        self.head_tilt = math.sin(self.frame_counter * 0.1) * 0.02
        
        # Breathing effect (subtle scaling)
        self.breathing_scale = 1.0 + math.sin(self.frame_counter * 0.05) * 0.01
    
    def _apply_animations(self) -> np.ndarray:
        """Apply all animations to the base face"""
        result = self.base_face.copy()
        
        # Apply breathing effect (subtle scaling)
        if self.breathing_scale != 1.0:
            result = self._apply_breathing(result)
        
        # Apply head tilt
        if abs(self.head_tilt) > 0.001:
            result = self._apply_head_tilt(result)
        
        # Apply mouth animation
        if self.mouth_open > 0.01:
            result = self._apply_mouth_animation(result)
        
        # Apply eye blinking
        if self.eye_blink > 0.01:
            result = self._apply_eye_blink(result)
        
        return result
    
    def _apply_breathing(self, image: np.ndarray) -> np.ndarray:
        """Apply subtle breathing effect"""
        height, width = image.shape[:2]
        
        # Create scaling matrix
        scale_matrix = cv2.getRotationMatrix2D((width/2, height/2), 0, self.breathing_scale)
        
        # Apply scaling
        result = cv2.warpAffine(image, scale_matrix, (width, height), 
                               borderMode=cv2.BORDER_REFLECT)
        
        return result
    
    def _apply_head_tilt(self, image: np.ndarray) -> np.ndarray:
        """Apply subtle head tilt"""
        height, width = image.shape[:2]
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 
                                                 self.head_tilt * 180 / math.pi, 1.0)
        
        # Apply rotation
        result = cv2.warpAffine(image, rotation_matrix, (width, height), 
                               borderMode=cv2.BORDER_REFLECT)
        
        return result
    
    def _apply_mouth_animation(self, image: np.ndarray) -> np.ndarray:
        """Apply mouth opening animation"""
        if self.mouth_region is None:
            return image
        
        result = image.copy()
        
        # Get mouth region
        x, y, w, h = (self.mouth_region['x'], self.mouth_region['y'], 
                     self.mouth_region['width'], self.mouth_region['height'])
        
        # Calculate mouth opening
        mouth_height = int(h * (0.3 + self.mouth_open * 0.7))
        mouth_y_offset = int((h - mouth_height) / 2)
        
        # Create mouth opening effect
        mouth_center_y = y + h // 2
        mouth_center_x = x + w // 2
        
        # Draw animated mouth (simple ellipse)
        mouth_width = int(w * 0.6)
        mouth_height_actual = int(mouth_height * 0.8)
        
        # Mouth color based on opening
        if self.mouth_open > 0.7:
            color = (0, 255, 0)  # Green for wide open
        elif self.mouth_open > 0.4:
            color = (255, 255, 0)  # Yellow for medium
        else:
            color = (255, 165, 0)  # Orange for slightly open
        
        # Draw mouth ellipse
        cv2.ellipse(result, (mouth_center_x, mouth_center_y), 
                   (mouth_width//2, mouth_height_actual//2), 
                   0, 0, 360, color, -1)
        
        return result
    
    def _apply_eye_blink(self, image: np.ndarray) -> np.ndarray:
        """Apply eye blinking animation"""
        if self.left_eye_region is None or self.right_eye_region is None:
            return image
        
        result = image.copy()
        
        # Apply blink to both eyes
        for eye_region in [self.left_eye_region, self.right_eye_region]:
            x, y, w, h = (eye_region['x'], eye_region['y'], 
                         eye_region['width'], eye_region['height'])
            
            # Calculate blink height
            blink_height = int(h * (1.0 - self.eye_blink * 0.8))
            
            # Create eyelid effect
            eye_center_x = x + w // 2
            eye_center_y = y + h // 2
            
            # Draw eyelid (dark line)
            cv2.line(result, (x, eye_center_y), (x + w, eye_center_y), 
                    (0, 0, 0), blink_height)
        
        return result
    
    def _add_debug_info(self, image: np.ndarray):
        """Add debug information to the image"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Add animation info
        info_text = f"Live2D: mouth={self.mouth_open:.2f}, blink={self.eye_blink:.2f}"
        cv2.putText(image, info_text, (10, 30), font, 0.6, (255, 255, 255), 2)
        
        # Add frame counter
        frame_text = f"Frame: {self.frame_counter}"
        cv2.putText(image, frame_text, (10, 60), font, 0.6, (255, 255, 255), 2)
        
        # Add status
        status_text = "🎭 Live2D Animation"
        cv2.putText(image, status_text, (10, 90), font, 0.6, (255, 255, 255), 2)
    
    def _bytes_to_audio_array(self, audio_data: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array"""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
            return audio_array
        except Exception as e:
            print(f"❌ LIVE2D: Error converting audio: {e}")
            return np.array([], dtype=np.float32) 