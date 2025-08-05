import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional
import logging

class MediaPipeAnimator:
    """MediaPipe-based face animator for Raspberry Pi"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_face = None
        self.frame_counter = 0
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Face mesh with high accuracy
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Mouth landmark indices (MediaPipe has 468 points)
        self.mouth_landmarks = [
            61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308
        ]
        
        self.logger.info("✅ MediaPipe animator initialized")
    
    def load_base_face(self, face_image_path: str) -> bool:
        """Load the base face image"""
        try:
            self.base_face = cv2.imread(face_image_path)
            if self.base_face is not None:
                print(f"✅ MEDIAPIPE: Loaded base face {self.base_face.shape}")
                
                # Convert to RGB for MediaPipe
                rgb_face = cv2.cvtColor(self.base_face, cv2.COLOR_BGR2RGB)
                
                # Detect face landmarks
                results = self.face_mesh.process(rgb_face)
                if results.multi_face_landmarks:
                    print(f"✅ MEDIAPIPE: Detected {len(results.multi_face_landmarks)} face(s)")
                    return True
                else:
                    print("❌ MEDIAPIPE: No face landmarks detected")
                    return False
            else:
                print(f"❌ MEDIAPIPE: Failed to load {face_image_path}")
                return False
        except Exception as e:
            print(f"❌ MEDIAPIPE: Error loading face: {e}")
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
                
                print(f"🎵 MEDIAPIPE: RMS={rms:.4f}, phoneme={phoneme_type}, mouth_open={mouth_open:.2f}")
            else:
                mouth_open = 0.3
                phoneme_type = "neutral"
                color = (255, 165, 0)
                print(f"🎵 MEDIAPIPE: No audio, mouth_open={mouth_open:.2f}")
            
            # Create animated face
            result = self.base_face.copy()
            
            # Convert to RGB for MediaPipe processing
            rgb_result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            
            # Detect landmarks on current frame
            results = self.face_mesh.process(rgb_result)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                
                # Get mouth landmarks
                mouth_points = []
                for idx in self.mouth_landmarks:
                    landmark = landmarks.landmark[idx]
                    x = int(landmark.x * result.shape[1])
                    y = int(landmark.y * result.shape[0])
                    mouth_points.append((x, y))
                
                if len(mouth_points) > 0:
                    # Calculate mouth center
                    mouth_center = np.mean(mouth_points, axis=0).astype(int)
                    
                    # Draw animated mouth overlay
                    mouth_radius = int(30 + (mouth_open * 50))  # 30-80 pixels
                    cv2.circle(result, tuple(mouth_center), mouth_radius, color, -1)
                    
                    # Add text label
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text_pos = (mouth_center[0] - 50, mouth_center[1] - mouth_radius - 20)
                    cv2.putText(result, phoneme_type.upper(), text_pos, font, 0.8, (255, 255, 255), 2)
                    
                    # Add frame counter
                    frame_text = f"Frame: {self.frame_counter}"
                    cv2.putText(result, frame_text, (10, 30), font, 0.7, (255, 255, 255), 2)
                    
                    # Add MediaPipe info
                    info_text = f"MediaPipe: {len(mouth_points)} points"
                    cv2.putText(result, info_text, (10, 60), font, 0.6, (255, 255, 255), 2)
            
            self.frame_counter += 1
            return result
            
        except Exception as e:
            print(f"❌ MEDIAPIPE ERROR: {e}")
            return self.base_face.copy() if self.base_face is not None else np.zeros((512, 512, 3), dtype=np.uint8)
    
    def _bytes_to_audio_array(self, audio_data: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array"""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
            return audio_array
        except Exception as e:
            print(f"❌ MEDIAPIPE: Error converting audio: {e}")
            return np.array([], dtype=np.float32)
    
    def cleanup(self):
        """Clean up MediaPipe resources"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close() 