import cv2
import numpy as np
import pygame
from typing import Tuple

class SimpleLipSync:
    """Simple lip-sync system that draws colored circles on the face"""
    
    def __init__(self, display_manager):
        self.display_manager = display_manager
        self.base_face = None
        self.frame_counter = 0
        
    def load_base_face(self, face_image_path: str) -> bool:
        """Load the base face image"""
        try:
            self.base_face = cv2.imread(face_image_path)
            if self.base_face is not None:
                print(f"✅ SIMPLE LIP SYNC: Loaded base face {self.base_face.shape}")
                return True
            else:
                print(f"❌ SIMPLE LIP SYNC: Failed to load {face_image_path}")
                return False
        except Exception as e:
            print(f"❌ SIMPLE LIP SYNC: Error loading face: {e}")
            return False
    
    def generate_face_for_audio_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Generate face with simple lip-sync based on audio"""
        try:
            if self.base_face is None:
                return np.zeros((512, 512, 3), dtype=np.uint8)
            
            # Simple audio analysis
            if len(audio_chunk) > 0:
                rms = np.sqrt(np.mean(audio_chunk**2))
                
                # Simple phoneme detection
                if rms > 0.1:
                    phoneme_type = "vowel"
                    circle_size = 100
                    circle_color = (0, 255, 0)  # Green
                elif rms > 0.05:
                    phoneme_type = "consonant"
                    circle_size = 70
                    circle_color = (255, 255, 0)  # Yellow
                elif rms > 0.02:
                    phoneme_type = "neutral"
                    circle_size = 50
                    circle_color = (255, 165, 0)  # Orange
                else:
                    phoneme_type = "closed"
                    circle_size = 30
                    circle_color = (255, 0, 0)  # Red
                
                print(f"🎵 SIMPLE AUDIO: RMS={rms:.4f}, phoneme={phoneme_type}, size={circle_size}")
            else:
                phoneme_type = "neutral"
                circle_size = 50
                circle_color = (255, 165, 0)  # Orange
                print(f"🎵 NO AUDIO: phoneme={phoneme_type}")
            
            # Create face with colored circle
            result = self.base_face.copy()
            
            # Draw circle in mouth area (center of image)
            center_x = result.shape[1] // 2
            center_y = result.shape[0] // 2 + 100  # Slightly below center for mouth
            
            cv2.circle(result, (center_x, center_y), circle_size, circle_color, -1)
            
            # Add text label
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_pos = (center_x - 50, center_y - circle_size - 20)
            cv2.putText(result, phoneme_type.upper(), text_pos, font, 1.0, (255, 255, 255), 2)
            
            # Add frame counter
            frame_text = f"Frame: {self.frame_counter}"
            cv2.putText(result, frame_text, (10, 30), font, 0.7, (255, 255, 255), 2)
            
            self.frame_counter += 1
            
            return result
            
        except Exception as e:
            print(f"❌ SIMPLE LIP SYNC ERROR: {e}")
            return self.base_face.copy() if self.base_face is not None else np.zeros((512, 512, 3), dtype=np.uint8)
    
    def _bytes_to_audio_array(self, audio_data: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array"""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
            return audio_array
        except Exception as e:
            print(f"❌ SIMPLE LIP SYNC: Error converting audio: {e}")
            return np.array([], dtype=np.float32) 