"""
Smart Mouth Animator - Actually Responds to Audio
Analyzes audio intensity and selects appropriate mouth shapes
Uses all available mouth images intelligently
"""

import cv2
import numpy as np
import logging
import math
from pathlib import Path
from typing import Dict, List

class SmartMouthAnimator:
    """Smart mouth animator that selects appropriate mouth shapes based on audio"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.mouth_shapes = {}
        self.current_shape = "mouth_closed"
        self.frame_counter = 0
        self.audio_history = []
        self.smoothing_factor = 0.3
        
        print("🎭 SMART MOUTH: Smart mouth animator initialized")
        self.logger.info("Smart mouth animator initialized")
    
    def load_mouth_shapes(self, faces_dir: str) -> bool:
        """Load all available mouth shape images"""
        try:
            faces_path = Path(faces_dir)
            
            # Define mouth shapes in order of openness
            shape_names = [
                "mouth_closed",
                "mouth_narrow", 
                "mouth_round",
                "mouth_open",
                "mouth_wide"
            ]
            
            # Load each mouth shape
            for shape_name in shape_names:
                shape_path = faces_path / f"{shape_name}.png"
                if shape_path.exists():
                    image = cv2.imread(str(shape_path))
                    if image is not None:
                        self.mouth_shapes[shape_name] = image
                        print(f"✅ SMART MOUTH: Loaded {shape_name} from {shape_path}")
                    else:
                        print(f"❌ SMART MOUTH: Failed to load {shape_path}")
                else:
                    print(f"⚠️ SMART MOUTH: {shape_path} not found")
            
            if not self.mouth_shapes:
                print("❌ SMART MOUTH: No mouth shapes loaded!")
                return False
            
            print(f"✅ SMART MOUTH: Loaded {len(self.mouth_shapes)} mouth shapes")
            return True
            
        except Exception as e:
            print(f"❌ SMART MOUTH: Error loading mouth shapes: {e}")
            return False
    
    def generate_face_for_audio_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Generate animated face by selecting appropriate mouth shape"""
        try:
            if not self.mouth_shapes:
                # Create fallback face
                fallback_face = np.ones((512, 512, 3), dtype=np.uint8) * 128
                cv2.putText(fallback_face, "Smart Mouth: No Shapes Loaded", (50, 256), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                return fallback_face
            
            # Analyze audio for mouth shape selection
            audio_intensity = self._analyze_audio_intensity(audio_chunk)
            
            # Select appropriate mouth shape
            selected_shape = self._select_mouth_shape(audio_intensity)
            
            # Get the selected mouth shape image
            result = self.mouth_shapes[selected_shape].copy()
            
            # Add debug info
            self._add_debug_info(result, audio_intensity, selected_shape)
            
            self.frame_counter += 1
            return result
            
        except Exception as e:
            print(f"❌ SMART MOUTH ERROR: {e}")
            # Return closed mouth as fallback
            if "mouth_closed" in self.mouth_shapes:
                return self.mouth_shapes["mouth_closed"].copy()
            else:
                fallback_face = np.ones((512, 512, 3), dtype=np.uint8) * 128
                cv2.putText(fallback_face, f"Smart Mouth Error: {str(e)[:30]}", (50, 256), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                return fallback_face
    
    def _analyze_audio_intensity(self, audio_chunk: np.ndarray) -> float:
        """Analyze audio chunk for intensity"""
        if len(audio_chunk) == 0:
            return 0.0
        
        # Calculate RMS for audio intensity
        rms = np.sqrt(np.mean(audio_chunk**2))
        
        # Add to history for smoothing
        self.audio_history.append(rms)
        if len(self.audio_history) > 5:  # Keep last 5 frames
            self.audio_history.pop(0)
        
        # Apply smoothing
        smoothed_intensity = np.mean(self.audio_history)
        
        # Add some natural variation
        variation = math.sin(self.frame_counter * 0.2) * 0.05
        
        intensity = smoothed_intensity + variation
        return max(0.0, min(1.0, intensity))
    
    def _select_mouth_shape(self, intensity: float) -> str:
        """Select appropriate mouth shape based on audio intensity"""
        # Define intensity thresholds for each shape
        thresholds = {
            "mouth_closed": 0.0,    # 0-20%
            "mouth_narrow": 0.2,    # 20-40%
            "mouth_round": 0.4,     # 40-60%
            "mouth_open": 0.6,      # 60-80%
            "mouth_wide": 0.8       # 80-100%
        }
        
        # Find the appropriate shape
        selected_shape = "mouth_closed"  # Default
        
        for shape_name, threshold in thresholds.items():
            if intensity >= threshold and shape_name in self.mouth_shapes:
                selected_shape = shape_name
        
        # Apply smoothing to avoid rapid switching
        if hasattr(self, 'current_shape') and self.current_shape in self.mouth_shapes:
            # Only switch if intensity difference is significant
            current_threshold = thresholds.get(self.current_shape, 0.0)
            if abs(intensity - current_threshold) > 0.15:  # 15% threshold
                self.current_shape = selected_shape
        else:
            self.current_shape = selected_shape
        
        return self.current_shape
    
    def _add_debug_info(self, image: np.ndarray, intensity: float, shape_name: str):
        """Add debug information to the image"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Add audio intensity
        intensity_text = f"Audio: {intensity:.2f}"
        cv2.putText(image, intensity_text, (10, 30), font, 0.6, (255, 255, 255), 2)
        
        # Add current shape
        shape_text = f"Shape: {shape_name}"
        cv2.putText(image, shape_text, (10, 60), font, 0.6, (255, 255, 255), 2)
        
        # Add frame counter
        frame_text = f"Frame: {self.frame_counter}"
        cv2.putText(image, frame_text, (10, 90), font, 0.6, (255, 255, 255), 2)
        
        # Add status
        status_text = "🎭 Smart Mouth Animation"
        cv2.putText(image, status_text, (10, 120), font, 0.6, (255, 255, 255), 2)
        
        # Add intensity bar
        bar_width = 200
        bar_height = 20
        bar_x, bar_y = 10, 150
        
        # Background bar
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        
        # Intensity bar
        fill_width = int(bar_width * intensity)
        if fill_width > 0:
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), (0, 255, 0), -1)
        
        # Border
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        
        # Intensity percentage
        percent_text = f"{int(intensity * 100)}%"
        cv2.putText(image, percent_text, (bar_x + bar_width + 10, bar_y + 15), font, 0.5, (255, 255, 255), 2)
    
    def _bytes_to_audio_array(self, audio_data: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array"""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
            return audio_array
        except Exception as e:
            print(f"❌ SMART MOUTH: Error converting audio: {e}")
            return np.array([], dtype=np.float32) 