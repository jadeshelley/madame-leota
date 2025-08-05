"""
Direct Mouth Animator - Simple and Effective
Works directly with the audio system and shows clear feedback
"""

import cv2
import numpy as np
import logging
import math
from pathlib import Path

class DirectMouthAnimator:
    """Direct mouth animator that works with the audio system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.mouth_shapes = {}
        self.current_shape = "mouth_closed"
        self.frame_counter = 0
        self.last_audio_intensity = 0.0
        
        print("🎭 DIRECT MOUTH: Direct mouth animator initialized")
        self.logger.info("Direct mouth animator initialized")
    
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
                        print(f"✅ DIRECT MOUTH: Loaded {shape_name} from {shape_path}")
                    else:
                        print(f"❌ DIRECT MOUTH: Failed to load {shape_path}")
                else:
                    print(f"⚠️ DIRECT MOUTH: {shape_path} not found")
            
            if not self.mouth_shapes:
                print("❌ DIRECT MOUTH: No mouth shapes loaded!")
                return False
            
            print(f"✅ DIRECT MOUTH: Loaded {len(self.mouth_shapes)} mouth shapes")
            return True
            
        except Exception as e:
            print(f"❌ DIRECT MOUTH: Error loading mouth shapes: {e}")
            return False
    
    def generate_face_for_audio_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Generate animated face by selecting appropriate mouth shape"""
        try:
            if not self.mouth_shapes:
                # Create fallback face
                fallback_face = np.ones((512, 512, 3), dtype=np.uint8) * 128
                cv2.putText(fallback_face, "Direct Mouth: No Shapes Loaded", (50, 256), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                return fallback_face
            
            # Analyze audio for mouth shape selection
            audio_intensity = self._analyze_audio_intensity(audio_chunk)
            
            # Select appropriate mouth shape
            selected_shape = self._select_mouth_shape(audio_intensity)
            
            # Get the selected mouth shape image
            result = self.mouth_shapes[selected_shape].copy()
            
            # Add comprehensive debug info
            self._add_debug_info(result, audio_chunk, audio_intensity, selected_shape)
            
            self.frame_counter += 1
            return result
            
        except Exception as e:
            print(f"❌ DIRECT MOUTH ERROR: {e}")
            # Return closed mouth as fallback
            if "mouth_closed" in self.mouth_shapes:
                return self.mouth_shapes["mouth_closed"].copy()
            else:
                fallback_face = np.ones((512, 512, 3), dtype=np.uint8) * 128
                cv2.putText(fallback_face, f"Direct Mouth Error: {str(e)[:30]}", (50, 256), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                return fallback_face
    
    def _analyze_audio_intensity(self, audio_chunk: np.ndarray) -> float:
        """Analyze audio chunk for intensity"""
        if len(audio_chunk) == 0:
            print(f"⚠️ DIRECT MOUTH: Empty audio chunk at frame {self.frame_counter}")
            return self.last_audio_intensity
        
        # Calculate RMS for audio intensity
        rms = np.sqrt(np.mean(audio_chunk**2))
        
        # Add some natural variation based on frame
        variation = math.sin(self.frame_counter * 0.1) * 0.1
        
        # Combine with previous intensity for smoothing
        intensity = (rms * 0.7 + self.last_audio_intensity * 0.3) + variation
        intensity = max(0.0, min(1.0, intensity))
        
        self.last_audio_intensity = intensity
        return intensity
    
    def _select_mouth_shape(self, intensity: float) -> str:
        """Select appropriate mouth shape based on audio intensity"""
        # Define intensity thresholds for each shape
        if intensity < 0.2:
            selected_shape = "mouth_closed"
        elif intensity < 0.4:
            selected_shape = "mouth_narrow"
        elif intensity < 0.6:
            selected_shape = "mouth_round"
        elif intensity < 0.8:
            selected_shape = "mouth_open"
        else:
            selected_shape = "mouth_wide"
        
        # Only switch if the shape exists and is different
        if selected_shape in self.mouth_shapes and selected_shape != self.current_shape:
            self.current_shape = selected_shape
            print(f"🎭 DIRECT MOUTH: Switched to {selected_shape} (intensity: {intensity:.2f})")
        
        return self.current_shape
    
    def _add_debug_info(self, image: np.ndarray, audio_chunk: np.ndarray, intensity: float, shape_name: str):
        """Add comprehensive debug information to the image"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Background for text
        cv2.rectangle(image, (5, 5), (400, 200), (0, 0, 0), -1)
        cv2.rectangle(image, (5, 5), (400, 200), (255, 255, 255), 2)
        
        # Add audio info
        audio_text = f"Audio Chunk: {len(audio_chunk)} samples"
        cv2.putText(image, audio_text, (10, 30), font, 0.5, (255, 255, 255), 1)
        
        # Add audio intensity
        intensity_text = f"Intensity: {intensity:.3f}"
        cv2.putText(image, intensity_text, (10, 55), font, 0.5, (255, 255, 255), 1)
        
        # Add current shape
        shape_text = f"Shape: {shape_name}"
        cv2.putText(image, shape_text, (10, 80), font, 0.5, (255, 255, 255), 1)
        
        # Add frame counter
        frame_text = f"Frame: {self.frame_counter}"
        cv2.putText(image, frame_text, (10, 105), font, 0.5, (255, 255, 255), 1)
        
        # Add status
        status_text = "🎭 Direct Mouth Animation"
        cv2.putText(image, status_text, (10, 130), font, 0.5, (255, 255, 255), 1)
        
        # Add intensity bar
        bar_width = 300
        bar_height = 25
        bar_x, bar_y = 10, 150
        
        # Background bar
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Intensity bar
        fill_width = int(bar_width * intensity)
        if fill_width > 0:
            # Color based on intensity
            if intensity < 0.3:
                color = (0, 255, 0)  # Green for low
            elif intensity < 0.6:
                color = (0, 255, 255)  # Yellow for medium
            else:
                color = (0, 0, 255)  # Red for high
            
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
        
        # Border
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        
        # Intensity percentage
        percent_text = f"{int(intensity * 100)}%"
        cv2.putText(image, percent_text, (bar_x + bar_width + 10, bar_y + 18), font, 0.6, (255, 255, 255), 2)
        
        # Add shape indicator
        shape_indicator = f"Current: {shape_name}"
        cv2.putText(image, shape_indicator, (10, 190), font, 0.5, (255, 255, 255), 1)
    
    def _bytes_to_audio_array(self, audio_data: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array"""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
            return audio_array
        except Exception as e:
            print(f"❌ DIRECT MOUTH: Error converting audio: {e}")
            return np.array([], dtype=np.float32) 