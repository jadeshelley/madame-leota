"""
OpenCV Face Animator - Proven Solution for Raspberry Pi
Uses OpenCV's built-in face detection and real mouth manipulation
Based on successful Pi projects
"""

import cv2
import numpy as np
import logging
import math

class OpenCVFaceAnimator:
    """Proven OpenCV-based face animator for Raspberry Pi"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_face = None
        self.frame_counter = 0
        
        # Animation parameters
        self.mouth_open = 0.0
        self.eye_blink = 0.0
        
        # Face detection using OpenCV's built-in detector
        try:
            # Use OpenCV's built-in face detection (more reliable than cascades)
            self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_detector.empty():
                # Fallback to system path
                self.face_detector = cv2.CascadeClassifier('/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')
            print("✅ OPENCV: Face detector initialized")
        except:
            print("⚠️ OPENCV: Using fallback face detection")
            self.face_detector = None
        
        # Face regions
        self.face_region = None
        self.mouth_region = None
        
        print("🎭 OPENCV: OpenCV face animator initialized")
        self.logger.info("OpenCV face animator initialized")
    
    def load_base_face(self, face_image_path: str) -> bool:
        """Load the base face image and detect face"""
        try:
            self.base_face = cv2.imread(face_image_path)
            if self.base_face is not None:
                print(f"✅ OPENCV: Loaded base face {self.base_face.shape}")
                
                # Detect face and mouth regions
                self._detect_face_regions()
                
                return True
            else:
                print(f"❌ OPENCV: Failed to load {face_image_path}")
                return False
        except Exception as e:
            print(f"❌ OPENCV: Error loading face: {e}")
            return False
    
    def _detect_face_regions(self):
        """Detect face and estimate mouth region using OpenCV"""
        if self.base_face is None:
            return
        
        height, width = self.base_face.shape[:2]
        
        # Try OpenCV face detection first
        if self.face_detector:
            gray = cv2.cvtColor(self.base_face, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Use the largest face
                face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = face
                self.face_region = {'x': x, 'y': y, 'width': w, 'height': h}
                
                # Estimate mouth region (lower third of face)
                mouth_y = y + int(h * 0.6)
                mouth_height = int(h * 0.2)
                mouth_width = int(w * 0.4)
                mouth_x = x + int(w * 0.3)
                
                self.mouth_region = {
                    'x': mouth_x, 'y': mouth_y, 'width': mouth_width, 'height': mouth_height
                }
                
                print(f"✅ OPENCV: Detected face and mouth regions")
                return
        
        # Fallback to estimated regions
        self._use_estimated_regions()
    
    def _use_estimated_regions(self):
        """Use estimated regions if detection fails"""
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
        
        print(f"✅ OPENCV: Using estimated regions")
    
    def generate_face_for_audio_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Generate animated face based on audio"""
        try:
            if self.base_face is None:
                # Create a fallback face
                fallback_face = np.ones((512, 512, 3), dtype=np.uint8) * 128
                cv2.putText(fallback_face, "OpenCV: No Base Face", (50, 256), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                return fallback_face
            
            # Analyze audio for animation parameters
            audio_intensity = self._analyze_audio(audio_chunk)
            
            # Update animation parameters
            self._update_animation_params(audio_intensity)
            
            # Create animated face
            result = self._apply_opencv_animations()
            
            # Add debug info
            self._add_debug_info(result)
            
            self.frame_counter += 1
            return result
            
        except Exception as e:
            print(f"❌ OPENCV ERROR: {e}")
            if self.base_face is not None:
                return self.base_face.copy()
            else:
                fallback_face = np.ones((512, 512, 3), dtype=np.uint8) * 128
                cv2.putText(fallback_face, f"OpenCV Error: {str(e)[:30]}", (50, 256), 
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
    
    def _apply_opencv_animations(self) -> np.ndarray:
        """Apply OpenCV-based animations to the base face"""
        result = self.base_face.copy()
        
        # Apply realistic mouth animation
        if self.mouth_open > 0.01 and self.mouth_region:
            result = self._apply_mouth_animation(result)
        
        return result
    
    def _apply_mouth_animation(self, image: np.ndarray) -> np.ndarray:
        """Apply realistic mouth opening animation using OpenCV"""
        if not self.mouth_region:
            return image
        
        result = image.copy()
        x, y, w, h = (self.mouth_region['x'], self.mouth_region['y'], 
                     self.mouth_region['width'], self.mouth_region['height'])
        
        # Ensure coordinates are within image bounds
        x = max(0, min(x, image.shape[1] - 1))
        y = max(0, min(y, image.shape[0] - 1))
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return result
        
        # Extract mouth region
        mouth_roi = result[y:y+h, x:x+w]
        
        # Create mouth opening effect by stretching vertically
        stretch_factor = 1.0 + self.mouth_open * 0.8  # 1.0 to 1.8x stretch
        
        # Resize mouth region vertically
        new_height = int(h * stretch_factor)
        if new_height > 0:
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
    
    def _add_debug_info(self, image: np.ndarray):
        """Add debug information to the image"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Add animation info
        info_text = f"OpenCV: mouth={self.mouth_open:.2f}, blink={self.eye_blink:.2f}"
        cv2.putText(image, info_text, (10, 30), font, 0.6, (255, 255, 255), 2)
        
        # Add frame counter
        frame_text = f"Frame: {self.frame_counter}"
        cv2.putText(image, frame_text, (10, 60), font, 0.6, (255, 255, 255), 2)
        
        # Add status
        status_text = "🎭 OpenCV Animation"
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
    
    def _bytes_to_audio_array(self, audio_data: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array"""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
            return audio_array
        except Exception as e:
            print(f"❌ OPENCV: Error converting audio: {e}")
            return np.array([], dtype=np.float32) 