"""
Face Animator for Madame Leota
Controls facial expressions, lip sync, and idle animations
"""

import cv2
import numpy as np
import asyncio
import logging
import time
import math
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from config import *

class FaceAnimator:
    def __init__(self, display_manager):
        self.logger = logging.getLogger(__name__)
        self.display_manager = display_manager
        
        # Animation state
        self.current_state = "idle"
        self.is_speaking = False
        self.idle_animation_running = False
        
        # Load face assets
        self.face_images = self._load_face_assets()
        
        # Animation timing
        self.animation_start_time = 0
        self.current_frame = 0
        
        # Mouth shape cache
        self.mouth_shapes = {}
        
        # Create base face if no assets found
        if not self.face_images:
            self._create_default_face()
        
        self.logger.info("Face Animator initialized")
    
    def _load_face_assets(self) -> Dict[str, np.ndarray]:
        """Load face image assets"""
        face_images = {}
        assets_path = Path(FACE_ASSETS_DIR)
        
        try:
            if assets_path.exists():
                for image_file in assets_path.glob("*.png"):
                    image_name = image_file.stem
                    image = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
                    if image is not None:
                        face_images[image_name] = image
                        self.logger.debug(f"Loaded face asset: {image_name}")
            
            return face_images
            
        except Exception as e:
            self.logger.error(f"Error loading face assets: {e}")
            return {}
    
    def _create_default_face(self):
        """Create a default Madame Leota face using OpenCV"""
        try:
            # Create base face image
            width, height = 400, 500
            face = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Colors
            skin_color = (220, 190, 180)  # Pale skin
            eye_color = (50, 150, 50)     # Green eyes
            hair_color = (40, 40, 40)     # Dark hair
            
            # Draw face oval
            cv2.ellipse(face, (width//2, height//2), (150, 180), 0, 0, 360, skin_color, -1)
            
            # Draw eyes
            eye_y = height//2 - 40
            left_eye_x = width//2 - 60
            right_eye_x = width//2 + 60
            
            # Eye shapes
            cv2.ellipse(face, (left_eye_x, eye_y), (25, 15), 0, 0, 360, (255, 255, 255), -1)
            cv2.ellipse(face, (right_eye_x, eye_y), (25, 15), 0, 0, 360, (255, 255, 255), -1)
            cv2.circle(face, (left_eye_x, eye_y), 12, eye_color, -1)
            cv2.circle(face, (right_eye_x, eye_y), 12, eye_color, -1)
            cv2.circle(face, (left_eye_x, eye_y), 6, (0, 0, 0), -1)
            cv2.circle(face, (right_eye_x, eye_y), 6, (0, 0, 0), -1)
            
            # Draw nose
            nose_points = np.array([
                [width//2 - 5, height//2],
                [width//2 + 5, height//2],
                [width//2, height//2 + 20]
            ], np.int32)
            cv2.fillPoly(face, [nose_points], skin_color)
            
            # Create different mouth shapes
            self._create_mouth_shapes(face, width, height)
            
            # Store base face
            self.face_images['base'] = face.copy()
            
        except Exception as e:
            self.logger.error(f"Error creating default face: {e}")
    
    def _create_mouth_shapes(self, base_face: np.ndarray, width: int, height: int):
        """Create different mouth shapes for lip sync"""
        mouth_center_x = width // 2
        mouth_center_y = height // 2 + 60
        
        # Define mouth shapes
        mouth_shapes = {
            'mouth_closed': {
                'points': [(mouth_center_x - 15, mouth_center_y), 
                          (mouth_center_x + 15, mouth_center_y)],
                'thickness': 3
            },
            'mouth_open': {
                'ellipse': ((mouth_center_x, mouth_center_y), (20, 30), 0),
                'color': (50, 50, 50)
            },
            'mouth_wide': {
                'ellipse': ((mouth_center_x, mouth_center_y), (35, 15), 0),
                'color': (50, 50, 50)
            },
            'mouth_round': {
                'ellipse': ((mouth_center_x, mouth_center_y), (15, 15), 0),
                'color': (50, 50, 50)
            },
            'mouth_narrow': {
                'ellipse': ((mouth_center_x, mouth_center_y), (25, 10), 0),
                'color': (50, 50, 50)
            }
        }
        
        # Generate face images with different mouth shapes
        for shape_name, shape_data in mouth_shapes.items():
            face_copy = base_face.copy()
            
            if 'ellipse' in shape_data:
                # Draw oval mouth
                cv2.ellipse(face_copy, shape_data['ellipse'][0], shape_data['ellipse'][1], 
                           shape_data['ellipse'][2], 0, 360, shape_data['color'], -1)
            elif 'points' in shape_data:
                # Draw line mouth
                cv2.line(face_copy, shape_data['points'][0], shape_data['points'][1], 
                        (100, 50, 50), shape_data['thickness'])
            
            self.face_images[shape_name] = face_copy
    
    async def animate_speaking(self, phonemes: List[Dict]):
        """Animate face according to phoneme timing for lip sync"""
        try:
            self.is_speaking = True
            self.current_state = "speaking"
            
            start_time = time.time()
            
            for phoneme_data in phonemes:
                if not self.is_speaking:
                    break
                
                # Calculate timing
                phoneme_start = phoneme_data['start_time'] / 1000.0  # Convert to seconds
                phoneme_duration = phoneme_data['duration'] / 1000.0
                mouth_shape = phoneme_data['mouth_shape']
                
                # Wait until phoneme start time
                current_time = time.time() - start_time
                if current_time < phoneme_start:
                    await asyncio.sleep(phoneme_start - current_time)
                
                # Display mouth shape
                await self._display_mouth_shape(mouth_shape, phoneme_duration)
            
            # Return to closed mouth
            await self._display_mouth_shape('mouth_closed', 0.2)
            
            self.is_speaking = False
            self.current_state = "idle"
            
        except Exception as e:
            self.logger.error(f"Speaking animation error: {e}")
            self.is_speaking = False
    
    async def _display_mouth_shape(self, mouth_shape: str, duration: float):
        """Display a specific mouth shape for given duration"""
        try:
            # Get face image with mouth shape
            face_image = self.face_images.get(mouth_shape, self.face_images.get('base'))
            
            if face_image is not None:
                # Add mystical effects
                enhanced_face = self._add_mystical_effects(face_image)
                
                # Display the face
                self.display_manager.clear_screen()
                self.display_manager.display_face(enhanced_face)
                self.display_manager.update_display()
            
            # Hold for duration
            if duration > 0:
                await asyncio.sleep(duration)
                
        except Exception as e:
            self.logger.error(f"Mouth shape display error: {e}")
    
    def start_idle_animation(self):
        """Start idle breathing animation"""
        if not self.idle_animation_running and not self.is_speaking:
            self.idle_animation_running = True
            asyncio.create_task(self._idle_animation_loop())
    
    def stop_idle_animation(self):
        """Stop idle animation"""
        self.idle_animation_running = False
    
    async def _idle_animation_loop(self):
        """Main idle animation loop with breathing effect"""
        try:
            while self.idle_animation_running and not self.is_speaking:
                current_time = time.time()
                
                # Breathing animation (slow sine wave)
                breath_cycle = math.sin(current_time * 0.5) * 0.1 + 1.0  # 0.9 to 1.1 scale
                
                # Eye blink occasionally
                blink = False
                if random.random() < 0.02:  # 2% chance per frame
                    blink = True
                
                # Get base face
                base_face = self.face_images.get('mouth_closed', self.face_images.get('base'))
                
                if base_face is not None:
                    # Apply breathing scale
                    scaled_face = self._apply_breathing_effect(base_face, breath_cycle)
                    
                    # Apply blink if needed
                    if blink:
                        scaled_face = self._apply_blink_effect(scaled_face)
                    
                    # Add mystical effects
                    enhanced_face = self._add_mystical_effects(scaled_face)
                    
                    # Display
                    self.display_manager.clear_screen()
                    self.display_manager.display_face(enhanced_face)
                    self.display_manager.update_display()
                
                await asyncio.sleep(1.0 / FPS)  # Maintain frame rate
                
        except Exception as e:
            self.logger.error(f"Idle animation error: {e}")
    
    def _apply_breathing_effect(self, face_image: np.ndarray, scale_factor: float) -> np.ndarray:
        """Apply subtle breathing scale effect"""
        try:
            height, width = face_image.shape[:2]
            
            # Create slight scale variation
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            # Resize
            scaled = cv2.resize(face_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # Pad or crop to original size
            if new_width != width or new_height != height:
                result = np.zeros_like(face_image)
                y_offset = (height - new_height) // 2
                x_offset = (width - new_width) // 2
                
                if y_offset >= 0 and x_offset >= 0:
                    result[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = scaled
                else:
                    # Crop if scaled image is larger
                    crop_y = max(0, -y_offset)
                    crop_x = max(0, -x_offset)
                    result = scaled[crop_y:crop_y+height, crop_x:crop_x+width]
                
                return result
            
            return scaled
            
        except Exception as e:
            self.logger.error(f"Breathing effect error: {e}")
            return face_image
    
    def _apply_blink_effect(self, face_image: np.ndarray) -> np.ndarray:
        """Apply eye blink effect"""
        # This is a simplified implementation
        # In a real version, you'd modify the eye regions specifically
        return face_image
    
    def _add_mystical_effects(self, face_image: np.ndarray) -> np.ndarray:
        """Add mystical glowing effects to the face"""
        try:
            # Add subtle glow
            glowing_face = self.display_manager.add_glow_effect(face_image, 0.2)
            
            # Add subtle color shift for mystical feel
            mystical_face = self._add_mystical_tint(glowing_face)
            
            return mystical_face
            
        except Exception as e:
            self.logger.error(f"Mystical effects error: {e}")
            return face_image
    
    def _add_mystical_tint(self, image: np.ndarray) -> np.ndarray:
        """Add mystical color tint"""
        try:
            # Create a subtle green/blue tint overlay
            tint = np.zeros_like(image)
            tint[:, :, 1] = 30  # Green channel
            tint[:, :, 0] = 10  # Blue channel
            
            # Blend with original
            mystical = cv2.addWeighted(image, 0.9, tint, 0.1, 0)
            
            return mystical
            
        except Exception as e:
            self.logger.error(f"Mystical tint error: {e}")
            return image
    
    def cleanup(self):
        """Cleanup animator resources"""
        self.idle_animation_running = False
        self.is_speaking = False
        self.logger.info("Face Animator cleaned up")

# Import random for idle animation
import random 