import cv2
import numpy as np
import dlib
import logging
from typing import Tuple, Optional
import os
from pathlib import Path

class CleanDlibAnimator:
    """Clean, simple dlib-based face animator for Raspberry Pi"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_face = None
        self.frame_counter = 0
        
        # Initialize dlib face detector and predictor
        try:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = self._load_predictor()
            print("✅ CLEAN DLIB: dlib initialized successfully")
        except Exception as e:
            print(f"❌ CLEAN DLIB: Failed to initialize dlib: {e}")
            self.detector = None
            self.predictor = None
        
        # Mouth landmark indices (dlib has 68 points)
        self.mouth_landmarks = list(range(48, 68))  # Points 48-67 are mouth
        
        self.logger.info("✅ Clean dlib animator initialized")
    
    def _load_predictor(self):
        """Load the facial landmark predictor"""
        try:
            # Try to find the predictor file
            predictor_paths = [
                "shape_predictor_68_face_landmarks.dat",
                "models/shape_predictor_68_face_landmarks.dat",
                "assets/shape_predictor_68_face_landmarks.dat"
            ]
            
            for path in predictor_paths:
                if os.path.exists(path):
                    print(f"✅ CLEAN DLIB: Found predictor at {path}")
                    return dlib.shape_predictor(path)
            
            # If not found, try to download it
            print("⚠️ CLEAN DLIB: Predictor not found, attempting download...")
            return self._download_predictor()
            
        except Exception as e:
            print(f"❌ CLEAN DLIB: Error loading predictor: {e}")
            return None
    
    def _download_predictor(self):
        """Download the facial landmark predictor"""
        try:
            import urllib.request
            
            url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            local_path = "shape_predictor_68_face_landmarks.dat.bz2"
            
            print(f"📥 CLEAN DLIB: Downloading predictor from {url}")
            urllib.request.urlretrieve(url, local_path)
            
            # Extract bz2 file
            import bz2
            with bz2.open(local_path, 'rb') as source, open('shape_predictor_68_face_landmarks.dat', 'wb') as target:
                target.write(source.read())
            
            # Clean up
            os.remove(local_path)
            
            print("✅ CLEAN DLIB: Predictor downloaded and extracted")
            return dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
            
        except Exception as e:
            print(f"❌ CLEAN DLIB: Failed to download predictor: {e}")
            return None
    
    def load_base_face(self, face_image_path: str) -> bool:
        """Load the base face image"""
        try:
            self.base_face = cv2.imread(face_image_path)
            if self.base_face is not None:
                print(f"✅ CLEAN DLIB: Loaded base face {self.base_face.shape}")
                
                # Test face detection
                if self.detector and self.predictor:
                    gray = cv2.cvtColor(self.base_face, cv2.COLOR_BGR2GRAY)
                    faces = self.detector(gray)
                    if len(faces) > 0:
                        print(f"✅ CLEAN DLIB: Detected {len(faces)} face(s)")
                        return True
                    else:
                        print("❌ CLEAN DLIB: No faces detected in image")
                        return False
                else:
                    print("⚠️ CLEAN DLIB: dlib not available, using fallback")
                    return True
            else:
                print(f"❌ CLEAN DLIB: Failed to load {face_image_path}")
                return False
        except Exception as e:
            print(f"❌ CLEAN DLIB: Error loading face: {e}")
            return False
    
    def generate_face_for_audio_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Generate animated face based on audio"""
        try:
            if self.base_face is None:
                return np.zeros((512, 512, 3), dtype=np.uint8)
            
            # Simple audio analysis
            if len(audio_chunk) > 0:
                rms = np.sqrt(np.mean(audio_chunk**2))
                
                # Map audio to mouth opening
                if rms > 0.1:
                    mouth_open = 0.8  # Wide open
                    phoneme_type = "vowel"
                    color = (0, 255, 0)  # Green
                elif rms > 0.05:
                    mouth_open = 0.6  # Medium open
                    phoneme_type = "consonant"
                    color = (255, 255, 0)  # Yellow
                elif rms > 0.02:
                    mouth_open = 0.4  # Slightly open
                    phoneme_type = "neutral"
                    color = (255, 165, 0)  # Orange
                else:
                    mouth_open = 0.1  # Nearly closed
                    phoneme_type = "closed"
                    color = (255, 0, 0)  # Red
                
                print(f"🎵 CLEAN DLIB: RMS={rms:.4f}, phoneme={phoneme_type}, mouth_open={mouth_open:.2f}")
            else:
                mouth_open = 0.3
                phoneme_type = "neutral"
                color = (255, 165, 0)
                print(f"🎵 CLEAN DLIB: No audio, mouth_open={mouth_open:.2f}")
            
            # Create animated face
            result = self.base_face.copy()
            
            # Try to detect face landmarks
            if self.detector and self.predictor:
                gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                faces = self.detector(gray)
                
                if len(faces) > 0:
                    # Get landmarks for first face
                    landmarks = self.predictor(gray, faces[0])
                    
                    # Get mouth landmarks
                    mouth_points = []
                    for i in self.mouth_landmarks:
                        point = landmarks.part(i)
                        mouth_points.append((point.x, point.y))
                    
                    if len(mouth_points) > 0:
                        # Calculate mouth center
                        mouth_center = np.mean(mouth_points, axis=0).astype(int)
                        
                        # Draw animated mouth overlay
                        mouth_radius = int(20 + (mouth_open * 40))  # 20-60 pixels
                        cv2.circle(result, tuple(mouth_center), mouth_radius, color, -1)
                        
                        # Add text label
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text_pos = (mouth_center[0] - 50, mouth_center[1] - mouth_radius - 20)
                        cv2.putText(result, phoneme_type.upper(), text_pos, font, 0.8, (255, 255, 255), 2)
                        
                        # Add dlib info
                        info_text = f"dlib: {len(mouth_points)} points"
                        cv2.putText(result, info_text, (10, 60), font, 0.6, (255, 255, 255), 2)
                    else:
                        # Fallback: draw in center
                        center_x = result.shape[1] // 2
                        center_y = result.shape[0] // 2 + 100
                        mouth_radius = int(20 + (mouth_open * 40))
                        cv2.circle(result, (center_x, center_y), mouth_radius, color, -1)
                        
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text_pos = (center_x - 50, center_y - mouth_radius - 20)
                        cv2.putText(result, phoneme_type.upper(), text_pos, font, 0.8, (255, 255, 255), 2)
                        
                        info_text = "dlib: fallback"
                        cv2.putText(result, info_text, (10, 60), font, 0.6, (255, 255, 255), 2)
                else:
                    # No face detected, use fallback
                    center_x = result.shape[1] // 2
                    center_y = result.shape[0] // 2 + 100
                    mouth_radius = int(20 + (mouth_open * 40))
                    cv2.circle(result, (center_x, center_y), mouth_radius, color, -1)
                    
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text_pos = (center_x - 50, center_y - mouth_radius - 20)
                    cv2.putText(result, phoneme_type.upper(), text_pos, font, 0.8, (255, 255, 255), 2)
                    
                    info_text = "dlib: no face"
                    cv2.putText(result, info_text, (10, 60), font, 0.6, (255, 255, 255), 2)
            else:
                # dlib not available, use simple fallback
                center_x = result.shape[1] // 2
                center_y = result.shape[0] // 2 + 100
                mouth_radius = int(20 + (mouth_open * 40))
                cv2.circle(result, (center_x, center_y), mouth_radius, color, -1)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_pos = (center_x - 50, center_y - mouth_radius - 20)
                cv2.putText(result, phoneme_type.upper(), text_pos, font, 0.8, (255, 255, 255), 2)
                
                info_text = "dlib: disabled"
                cv2.putText(result, info_text, (10, 60), font, 0.6, (255, 255, 255), 2)
            
            # Add frame counter
            font = cv2.FONT_HERSHEY_SIMPLEX
            frame_text = f"Frame: {self.frame_counter}"
            cv2.putText(result, frame_text, (10, 30), font, 0.7, (255, 255, 255), 2)
            
            self.frame_counter += 1
            return result
            
        except Exception as e:
            print(f"❌ CLEAN DLIB ERROR: {e}")
            return self.base_face.copy() if self.base_face is not None else np.zeros((512, 512, 3), dtype=np.uint8)
    
    def _bytes_to_audio_array(self, audio_data: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array"""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
            return audio_array
        except Exception as e:
            print(f"❌ CLEAN DLIB: Error converting audio: {e}")
            return np.array([], dtype=np.float32) 