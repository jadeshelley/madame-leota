"""
Realistic Face Animator for Madame Leota
Actually manipulates the real mouth and eyes in the face image
"""

import cv2
import numpy as np
import logging
import os
from pathlib import Path
from typing import Optional, Tuple
import math

class RealisticFaceAnimator:
    """Realistic face animator that manipulates actual face features"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_face = None
        self.frame_counter = 0
        
        # Animation parameters
        self.mouth_open = 0.0
        self.eye_blink = 0.0
        self.head_tilt = 0.0
        self.breathing_scale = 1.0
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Face regions (will be detected)
        self.face_region = None
        self.mouth_region = None
        self.left_eye_region = None
        self.right_eye_region = None
        
        print("🎭 REALISTIC: Realistic face animator initialized")
        self.logger.info("Realistic face animator initialized")
    
    def load_base_face(self, face_image_path: str) -> bool:
        """Load the base face image and detect face features"""
        try:
            self.base_face = cv2.imread(face_image_path)
            if self.base_face is not None:
                print(f"✅ REALISTIC: Loaded base face {self.base_face.shape}")
                
                # Detect face features
                self._detect_face_features()
                
                return True
            else:
                print(f"❌ REALISTIC: Failed to load {face_image_path}")
                return False
        except Exception as e:
            print(f"❌ REALISTIC: Error loading face: {e}")
            return False
    
    def _detect_face_features(self):
        """Detect face, eyes, and mouth regions"""
        if self.base_face is None:
            return
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(self.base_face, cv2.COLOR_BGR2GRAY)
        
        # Detect face
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Use the largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            self.face_region = {'x': x, 'y': y, 'width': w, 'height': h}
            
            # Detect eyes in face region
            face_roi = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 4)
            
            if len(eyes) >= 2:
                # Sort eyes by x position (left to right)
                eyes = sorted(eyes, key=lambda e: e[0])
                
                # Left eye
                ex, ey, ew, eh = eyes[0]
                self.left_eye_region = {
                    'x': x + ex, 'y': y + ey, 'width': ew, 'height': eh
                }
                
                # Right eye
                ex, ey, ew, eh = eyes[1]
                self.right_eye_region = {
                    'x': x + ex, 'y': y + ey, 'width': ew, 'height': eh
                }
            
            # Estimate mouth region (below eyes, in lower third of face)
            mouth_y = y + int(h * 0.6)
            mouth_height = int(h * 0.2)
            mouth_width = int(w * 0.4)
            mouth_x = x + int(w * 0.3)
            
            self.mouth_region = {
                'x': mouth_x, 'y': mouth_y, 'width': mouth_width, 'height': mouth_height
            }
            
            print(f"✅ REALISTIC: Detected face features - face: {self.face_region}, mouth: {self.mouth_region}")
            if self.left_eye_region and self.right_eye_region:
                print(f"✅ REALISTIC: Eyes detected - left: {self.left_eye_region}, right: {self.right_eye_region}")
        else:
            print("⚠️ REALISTIC: No face detected, using estimated regions")
            self._use_estimated_regions()
    
    def _use_estimated_regions(self):
        """Use estimated regions if face detection fails"""
        height, width = self.base_face.shape[:2]
        
        # Face region (most of the image)
        self.face_region = {
            'x': int(width * 0.1), 'y': int(height * 0.1),
            'width': int(width * 0.8), 'height': int(height * 0.8)
        }
        
        # Mouth region (lower third)
        mouth_y = int(height * 0.6)
        mouth_height = int(height * 0.2)
        self.mouth_region = {
            'x': int(width * 0.3), 'y': mouth_y,
            'width': int(width * 0.4), 'height': mouth_height
        }
        
        # Eye regions (upper third)
        eye_y = int(height * 0.25)
        eye_height = int(height * 0.15)
        
        self.left_eye_region = {
            'x': int(width * 0.25), 'y': eye_y,
            'width': int(width * 0.15), 'height': eye_height
        }
        
        self.right_eye_region = {
            'x': int(width * 0.6), 'y': eye_y,
            'width': int(width * 0.15), 'height': eye_height
        }
    
    def generate_face_for_audio_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Generate animated face based on audio"""
        try:
            if self.base_face is None:
                # Create a fallback face instead of black screen
                fallback_face = np.ones((512, 512, 3), dtype=np.uint8) * 128
                cv2.putText(fallback_face, "Realistic: No Base Face", (50, 256), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                return fallback_face
            
            # Analyze audio for animation parameters
            audio_intensity = self._analyze_audio(audio_chunk)
            
            # Update animation parameters
            self._update_animation_params(audio_intensity)
            
            # Create animated face
            result = self._apply_realistic_animations()
            
            # Add debug info
            self._add_debug_info(result)
            
            self.frame_counter += 1
            return result
            
        except Exception as e:
            print(f"❌ REALISTIC ERROR: {e}")
            if self.base_face is not None:
                return self.base_face.copy()
            else:
                fallback_face = np.ones((512, 512, 3), dtype=np.uint8) * 128
                cv2.putText(fallback_face, f"Realistic Error: {str(e)[:30]}", (50, 256), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                return fallback_face
    
    def _analyze_audio(self, audio_chunk: np.ndarray) -> float:
        """Analyze audio chunk for animation intensity"""
        if len(audio_chunk) == 0:
            return 0.0
        
        # Calculate RMS for audio intensity
        rms = np.sqrt(np.mean(audio_chunk**2))
        
        # Add frame-based variation for natural movement
        frame_variation = math.sin(self.frame_counter * 0.3) * 0.1
        intensity = rms + frame_variation
        
        return max(0.0, min(1.0, intensity))
    
    def _update_animation_params(self, audio_intensity: float):
        """Update animation parameters based on audio"""
        # Mouth opening (responds to audio intensity)
        target_mouth = audio_intensity * 0.8
        self.mouth_open = self.mouth_open * 0.8 + target_mouth * 0.2
        
        # Eye blinking (independent of audio)
        blink_cycle = (self.frame_counter % 60) / 60.0
        if blink_cycle > 0.95:
            self.eye_blink = 1.0
        else:
            self.eye_blink = max(0.0, self.eye_blink - 0.1)
        
        # Head tilt (subtle movement)
        self.head_tilt = math.sin(self.frame_counter * 0.1) * 0.02
        
        # Breathing effect (subtle scaling)
        self.breathing_scale = 1.0 + math.sin(self.frame_counter * 0.05) * 0.01
    
    def _apply_realistic_animations(self) -> np.ndarray:
        """Apply realistic animations to the base face"""
        result = self.base_face.copy()
        
        # Apply breathing effect
        if self.breathing_scale != 1.0:
            result = self._apply_breathing(result)
        
        # Apply head tilt
        if abs(self.head_tilt) > 0.001:
            result = self._apply_head_tilt(result)
        
        # Apply realistic mouth animation
        if self.mouth_open > 0.01 and self.mouth_region:
            result = self._apply_realistic_mouth_animation(result)
        
        # Apply realistic eye blinking
        if self.eye_blink > 0.01 and self.left_eye_region and self.right_eye_region:
            result = self._apply_realistic_eye_blink(result)
        
        return result
    
    def _apply_breathing(self, image: np.ndarray) -> np.ndarray:
        """Apply subtle breathing effect"""
        height, width = image.shape[:2]
        scale_matrix = cv2.getRotationMatrix2D((width/2, height/2), 0, self.breathing_scale)
        result = cv2.warpAffine(image, scale_matrix, (width, height), 
                               borderMode=cv2.BORDER_REFLECT)
        return result
    
    def _apply_head_tilt(self, image: np.ndarray) -> np.ndarray:
        """Apply subtle head tilt"""
        height, width = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 
                                                 self.head_tilt * 180 / math.pi, 1.0)
        result = cv2.warpAffine(image, rotation_matrix, (width, height), 
                               borderMode=cv2.BORDER_REFLECT)
        return result
    
    def _apply_realistic_mouth_animation(self, image: np.ndarray) -> np.ndarray:
        """Apply realistic mouth opening animation"""
        if not self.mouth_region:
            return image
        
        result = image.copy()
        x, y, w, h = (self.mouth_region['x'], self.mouth_region['y'], 
                     self.mouth_region['width'], self.mouth_region['height'])
        
        # Extract mouth region
        mouth_roi = result[y:y+h, x:x+w]
        
        # Create mouth opening effect by stretching vertically
        stretch_factor = 1.0 + self.mouth_open * 0.5  # 1.0 to 1.5x stretch
        
        # Resize mouth region vertically
        new_height = int(h * stretch_factor)
        stretched_mouth = cv2.resize(mouth_roi, (w, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Calculate new position to keep center aligned
        y_offset = int((new_height - h) / 2)
        new_y = max(0, y - y_offset)
        new_y_end = min(result.shape[0], new_y + new_height)
        
        # Blend the stretched mouth back into the image
        if new_y_end - new_y == new_height and x + w <= result.shape[1]:
            # Create a mask for smooth blending
            mask = np.ones((new_height, w, 3), dtype=np.float32)
            mask = cv2.GaussianBlur(mask, (21, 21), 0)
            
            # Blend the stretched mouth
            roi = result[new_y:new_y_end, x:x+w]
            blended = (stretched_mouth * mask + roi * (1 - mask)).astype(np.uint8)
            result[new_y:new_y_end, x:x+w] = blended
        
        return result
    
    def _apply_realistic_eye_blink(self, image: np.ndarray) -> np.ndarray:
        """Apply realistic eye blinking animation"""
        if not self.left_eye_region or not self.right_eye_region:
            return image
        
        result = image.copy()
        
        # Apply blink to both eyes
        for eye_region in [self.left_eye_region, self.right_eye_region]:
            x, y, w, h = (eye_region['x'], eye_region['y'], 
                         eye_region['width'], eye_region['height'])
            
            # Extract eye region
            eye_roi = result[y:y+h, x:x+w]
            
            # Create eyelid effect by compressing the eye vertically
            compression_factor = 1.0 - self.eye_blink * 0.8  # 1.0 to 0.2x compression
            
            # Resize eye region vertically
            new_height = int(h * compression_factor)
            if new_height > 0:
                compressed_eye = cv2.resize(eye_roi, (w, new_height), interpolation=cv2.INTER_LINEAR)
                
                # Calculate new position to keep center aligned
                y_offset = int((h - new_height) / 2)
                new_y = y + y_offset
                
                # Blend the compressed eye back into the image
                if new_y + new_height <= result.shape[0] and x + w <= result.shape[1]:
                    # Create a mask for smooth blending
                    mask = np.ones((new_height, w, 3), dtype=np.float32)
                    mask = cv2.GaussianBlur(mask, (15, 15), 0)
                    
                    # Blend the compressed eye
                    roi = result[new_y:new_y+new_height, x:x+w]
                    blended = (compressed_eye * mask + roi * (1 - mask)).astype(np.uint8)
                    result[new_y:new_y+new_height, x:x+w] = blended
        
        return result
    
    def _add_debug_info(self, image: np.ndarray):
        """Add debug information to the image"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Add animation info
        info_text = f"Realistic: mouth={self.mouth_open:.2f}, blink={self.eye_blink:.2f}"
        cv2.putText(image, info_text, (10, 30), font, 0.6, (255, 255, 255), 2)
        
        # Add frame counter
        frame_text = f"Frame: {self.frame_counter}"
        cv2.putText(image, frame_text, (10, 60), font, 0.6, (255, 255, 255), 2)
        
        # Add status
        status_text = "🎭 Realistic Animation"
        cv2.putText(image, status_text, (10, 90), font, 0.6, (255, 255, 255), 2)
        
        # Draw detection boxes if available
        if self.face_region:
            x, y, w, h = (self.face_region['x'], self.face_region['y'], 
                         self.face_region['width'], self.face_region['height'])
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        if self.mouth_region:
            x, y, w, h = (self.mouth_region['x'], self.mouth_region['y'], 
                         self.mouth_region['width'], self.mouth_region['height'])
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        if self.left_eye_region:
            x, y, w, h = (self.left_eye_region['x'], self.left_eye_region['y'], 
                         self.left_eye_region['width'], self.left_eye_region['height'])
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        if self.right_eye_region:
            x, y, w, h = (self.right_eye_region['x'], self.right_eye_region['y'], 
                         self.right_eye_region['width'], self.right_eye_region['height'])
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    def _bytes_to_audio_array(self, audio_data: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array"""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
            return audio_array
        except Exception as e:
            print(f"❌ REALISTIC: Error converting audio: {e}")
            return np.array([], dtype=np.float32) 