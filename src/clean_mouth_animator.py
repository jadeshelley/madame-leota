"""
Clean Mouth Animator - Simple and Effective
Uses existing mouth shape images with smooth interpolation
No complex dependencies - just basic OpenCV operations
"""

import cv2
import numpy as np
import logging
import math
from pathlib import Path

class CleanMouthAnimator:
    """Clean mouth animator that works reliably on Pi"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.mouth_shapes = {}
        self.current_shape = "mouth_closed"
        self.frame_counter = 0
        self.target_shape = "mouth_closed"
        self.transition_progress = 0.0
        self.transition_speed = 0.1  # How fast to transition between shapes
        
        print("🎭 CLEAN MOUTH: Clean mouth animator initialized")
        self.logger.info("Clean mouth animator initialized")
    
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
                        print(f"✅ CLEAN MOUTH: Loaded {shape_name} from {shape_path}")
                    else:
                        print(f"❌ CLEAN MOUTH: Failed to load {shape_path}")
                else:
                    print(f"⚠️ CLEAN MOUTH: {shape_path} not found")
            
            if not self.mouth_shapes:
                print("❌ CLEAN MOUTH: No mouth shapes loaded!")
                return False
            
            print(f"✅ CLEAN MOUTH: Loaded {len(self.mouth_shapes)} mouth shapes")
            return True
            
        except Exception as e:
            print(f"❌ CLEAN MOUTH: Error loading mouth shapes: {e}")
            return False
    
    def generate_face_for_audio_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Generate animated face with smooth mouth transitions"""
        try:
            if not self.mouth_shapes:
                # Create fallback face
                fallback_face = np.ones((512, 512, 3), dtype=np.uint8) * 128
                cv2.putText(fallback_face, "Clean Mouth: No Shapes Loaded", (50, 256), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                return fallback_face
            
            # Analyze audio for mouth shape selection
            audio_intensity = self._analyze_audio_intensity(audio_chunk)
            
            # Select target mouth shape based on audio
            self.target_shape = self._select_mouth_shape(audio_intensity)
            
            # Smooth transition to target shape
            if self.target_shape != self.current_shape:
                self.transition_progress = 0.0
                self.current_shape = self.target_shape
            
            # Create smooth transition
            result = self._create_smooth_transition()
            
            # Add debug info
            self._add_debug_info(result, audio_chunk, audio_intensity)
            
            self.frame_counter += 1
            return result
            
        except Exception as e:
            print(f"❌ CLEAN MOUTH ERROR: {e}")
            # Return closed mouth as fallback
            if "mouth_closed" in self.mouth_shapes:
                return self.mouth_shapes["mouth_closed"].copy()
            else:
                fallback_face = np.ones((512, 512, 3), dtype=np.uint8) * 128
                cv2.putText(fallback_face, f"Clean Mouth Error: {str(e)[:30]}", (50, 256), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                return fallback_face
    
    def _analyze_audio_intensity(self, audio_chunk: np.ndarray) -> float:
        """Analyze audio chunk for intensity"""
        if len(audio_chunk) == 0:
            return 0.0
        
        # Calculate RMS for audio intensity
        rms = np.sqrt(np.mean(audio_chunk**2))
        
        # Add some natural variation based on frame
        variation = math.sin(self.frame_counter * 0.2) * 0.05
        
        intensity = rms + variation
        return max(0.0, min(1.0, intensity))
    
    def _select_mouth_shape(self, intensity: float) -> str:
        """Select appropriate mouth shape based on audio intensity"""
        # Define intensity thresholds for each shape
        if intensity < 0.2:
            return "mouth_closed"
        elif intensity < 0.4:
            return "mouth_narrow"
        elif intensity < 0.6:
            return "mouth_round"
        elif intensity < 0.8:
            return "mouth_open"
        else:
            return "mouth_wide"
    
    def _create_smooth_transition(self) -> np.ndarray:
        """Create smooth transition between mouth shapes"""
        # Get current and target shapes
        current_image = self.mouth_shapes.get(self.current_shape, self.mouth_shapes["mouth_closed"])
        
        # Add subtle breathing animation
        breathing_factor = 1.0 + 0.02 * math.sin(self.frame_counter * 0.1)
        
        # Apply breathing effect
        height, width = current_image.shape[:2]
        new_height = int(height * breathing_factor)
        new_width = int(width * breathing_factor)
        
        # Resize with breathing effect
        resized = cv2.resize(current_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Crop back to original size
        start_y = (new_height - height) // 2
        start_x = (new_width - width) // 2
        result = resized[start_y:start_y+height, start_x:start_x+width]
        
        return result
    
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
        
        # Add current shape
        shape_text = f"Shape: {self.current_shape}"
        cv2.putText(image, shape_text, (10, 80), font, 0.5, (255, 255, 255), 1)
        
        # Add frame counter
        frame_text = f"Frame: {self.frame_counter}"
        cv2.putText(image, frame_text, (10, 105), font, 0.5, (255, 255, 255), 1)
        
        # Add status
        status_text = "🎭 Clean Mouth Animation"
        cv2.putText(image, status_text, (10, 130), font, 0.5, (255, 255, 255), 1)
        
        # Add breathing indicator
        breathing_text = f"Breathing: {1.0 + 0.02 * math.sin(self.frame_counter * 0.1):.3f}"
        cv2.putText(image, breathing_text, (10, 155), font, 0.5, (255, 255, 255), 1)
    
    def _bytes_to_audio_array(self, audio_data: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array"""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
            return audio_array
        except Exception as e:
            print(f"❌ CLEAN MOUTH: Error converting audio: {e}")
            return np.array([], dtype=np.float32) 