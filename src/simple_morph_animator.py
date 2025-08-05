"""
Simple Morph Animator - Guaranteed to Work on Pi
Uses two base images and cross-fades between them based on audio
No complex detection, no AI, just simple image blending
"""

import cv2
import numpy as np
import logging
import math

class SimpleMorphAnimator:
    """Simple morphing animator that cross-fades between two images"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.mouth_closed = None
        self.mouth_open = None
        self.frame_counter = 0
        
        print("🎭 SIMPLE MORPH: Simple morphing animator initialized")
        self.logger.info("Simple morphing animator initialized")
    
    def load_base_face(self, face_image_path: str) -> bool:
        """Load the base face images for morphing"""
        try:
            # Load the closed mouth image
            self.mouth_closed = cv2.imread(face_image_path)
            if self.mouth_closed is None:
                print(f"❌ SIMPLE MORPH: Failed to load {face_image_path}")
                return False
            
            print(f"✅ SIMPLE MORPH: Loaded closed mouth image {self.mouth_closed.shape}")
            
            # Create an open mouth version by stretching the mouth region
            self.mouth_open = self._create_open_mouth_version()
            
            return True
            
        except Exception as e:
            print(f"❌ SIMPLE MORPH: Error loading face: {e}")
            return False
    
    def _create_open_mouth_version(self):
        """Create an open mouth version of the base image"""
        if self.mouth_closed is None:
            return None
        
        # Copy the closed mouth image
        open_mouth = self.mouth_closed.copy()
        height, width = open_mouth.shape[:2]
        
        # Define mouth region (lower third of image)
        mouth_y = int(height * 0.6)
        mouth_height = int(height * 0.2)
        mouth_width = int(width * 0.4)
        mouth_x = int(width * 0.3)
        
        # Extract mouth region
        mouth_roi = open_mouth[mouth_y:mouth_y+mouth_height, mouth_x:mouth_x+mouth_width]
        
        # Stretch the mouth region vertically to simulate opening
        stretch_factor = 1.5  # 50% taller
        new_height = int(mouth_height * stretch_factor)
        stretched_mouth = cv2.resize(mouth_roi, (mouth_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Calculate new position to keep center aligned
        y_offset = int((new_height - mouth_height) / 2)
        new_y = max(0, mouth_y - y_offset)
        new_y_end = min(height, new_y + new_height)
        
        # Blend the stretched mouth back into the image
        if new_y_end - new_y == new_height and mouth_x + mouth_width <= width:
            # Create a mask for smooth blending
            mask = np.ones((new_height, mouth_width, 3), dtype=np.float32)
            mask = cv2.GaussianBlur(mask, (21, 21), 0)
            
            # Blend the stretched mouth
            roi = open_mouth[new_y:new_y_end, mouth_x:mouth_x+mouth_width]
            blended = (stretched_mouth * mask + roi * (1 - mask)).astype(np.uint8)
            open_mouth[new_y:new_y_end, mouth_x:mouth_x+mouth_width] = blended
        
        print(f"✅ SIMPLE MORPH: Created open mouth version")
        return open_mouth
    
    def generate_face_for_audio_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Generate animated face by morphing between closed and open mouth"""
        try:
            if self.mouth_closed is None or self.mouth_open is None:
                # Create a fallback face
                fallback_face = np.ones((512, 512, 3), dtype=np.uint8) * 128
                cv2.putText(fallback_face, "Simple Morph: No Base Images", (50, 256), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                return fallback_face
            
            # Analyze audio for morphing intensity
            audio_intensity = self._analyze_audio(audio_chunk)
            
            # Create morphed image
            result = self._morph_images(audio_intensity)
            
            # Add debug info
            self._add_debug_info(result, audio_intensity)
            
            self.frame_counter += 1
            return result
            
        except Exception as e:
            print(f"❌ SIMPLE MORPH ERROR: {e}")
            if self.mouth_closed is not None:
                return self.mouth_closed.copy()
            else:
                fallback_face = np.ones((512, 512, 3), dtype=np.uint8) * 128
                cv2.putText(fallback_face, f"Simple Morph Error: {str(e)[:30]}", (50, 256), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                return fallback_face
    
    def _analyze_audio(self, audio_chunk: np.ndarray) -> float:
        """Analyze audio chunk for morphing intensity"""
        if len(audio_chunk) == 0:
            return 0.0
        
        # Calculate RMS for audio intensity
        rms = np.sqrt(np.mean(audio_chunk**2))
        
        # Add frame-based variation for natural movement
        frame_variation = math.sin(self.frame_counter * 0.3) * 0.1
        intensity = rms + frame_variation
        
        return max(0.0, min(1.0, intensity))
    
    def _morph_images(self, intensity: float) -> np.ndarray:
        """Morph between closed and open mouth images"""
        # Convert images to float for blending
        closed_float = self.mouth_closed.astype(np.float32)
        open_float = self.mouth_open.astype(np.float32)
        
        # Blend the images based on intensity
        # intensity = 0.0 -> fully closed mouth
        # intensity = 1.0 -> fully open mouth
        morphed = closed_float * (1.0 - intensity) + open_float * intensity
        
        return morphed.astype(np.uint8)
    
    def _add_debug_info(self, image: np.ndarray, intensity: float):
        """Add debug information to the image"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Add animation info
        info_text = f"Simple Morph: intensity={intensity:.2f}"
        cv2.putText(image, info_text, (10, 30), font, 0.6, (255, 255, 255), 2)
        
        # Add frame counter
        frame_text = f"Frame: {self.frame_counter}"
        cv2.putText(image, frame_text, (10, 60), font, 0.6, (255, 255, 255), 2)
        
        # Add status
        status_text = "🎭 Simple Morph Animation"
        cv2.putText(image, status_text, (10, 90), font, 0.6, (255, 255, 255), 2)
        
        # Add morph indicator
        morph_text = f"Morph: {int(intensity * 100)}% open"
        cv2.putText(image, morph_text, (10, 120), font, 0.6, (255, 255, 255), 2)
    
    def _bytes_to_audio_array(self, audio_data: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array"""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
            return audio_array
        except Exception as e:
            print(f"❌ SIMPLE MORPH: Error converting audio: {e}")
            return np.array([], dtype=np.float32) 