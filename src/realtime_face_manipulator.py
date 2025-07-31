"""
Real-Time Face Manipulator for Madame Leota
Uses MediaPipe for facial landmarks and real-time mouth manipulation
Optimized for Raspberry Pi performance
"""

import cv2
import numpy as np
import mediapipe as mp
import logging
import asyncio
import time
from typing import List, Dict, Tuple, Optional
from config import *

class RealtimeFaceManipulator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Base face image (the realistic face)
        self.base_face = None
        self.face_landmarks = None
        self.face_dimensions = None
        
        # Mouth landmark indices (MediaPipe face mesh)
        self.mouth_landmarks = [
            # Outer lip landmarks
            61, 146, 91, 181, 84, 17, 314, 405, 320, 375, 321, 308,
            # Inner lip landmarks  
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 415, 310,
            # Additional mouth points for better coverage
            13, 82, 81, 80, 78, 191, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324
        ]
        
        # Phoneme to mouth transformation mapping
        self.phoneme_transforms = self._create_phoneme_transforms()
        
        self.logger.info("Real-time Face Manipulator initialized")
    
    def load_base_face(self, face_image_path: str) -> bool:
        """Load the base face image for manipulation"""
        try:
            # Load the main face image (mouth_closed.png)
            self.base_face = cv2.imread(face_image_path)
            if self.base_face is None:
                self.logger.error(f"Could not load base face: {face_image_path}")
                return False
            
            # Convert to RGB for MediaPipe
            rgb_face = cv2.cvtColor(self.base_face, cv2.COLOR_BGR2RGB)
            
            # Detect facial landmarks
            results = self.face_mesh.process(rgb_face)
            
            if results.multi_face_landmarks:
                self.face_landmarks = results.multi_face_landmarks[0]
                h, w = self.base_face.shape[:2]
                self.face_dimensions = (w, h)
                
                self.logger.info("Base face loaded and landmarks detected")
                return True
            else:
                self.logger.error("No face landmarks detected in base image")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading base face: {e}")
            return False
    
    def _create_phoneme_transforms(self) -> Dict:
        """Create mouth transformation parameters for different phonemes"""
        return {
            # Vowels - mouth opening variations
            'mouth_open': {
                'vertical_scale': 1.8,    # Stretch mouth vertically
                'horizontal_scale': 1.0,
                'jaw_drop': 15,           # Lower jaw position
                'lip_separation': 12
            },
            'mouth_wide': {
                'vertical_scale': 0.8,
                'horizontal_scale': 1.6,  # Stretch mouth horizontally  
                'jaw_drop': 5,
                'lip_separation': 6
            },
            'mouth_round': {
                'vertical_scale': 1.3,
                'horizontal_scale': 0.8,  # Compress horizontally
                'jaw_drop': 8,
                'lip_separation': 10,
                'pucker': True            # Round the lips
            },
            'mouth_narrow': {
                'vertical_scale': 1.1,
                'horizontal_scale': 0.9,
                'jaw_drop': 3,
                'lip_separation': 4
            },
            'mouth_closed': {
                'vertical_scale': 1.0,
                'horizontal_scale': 1.0,
                'jaw_drop': 0,
                'lip_separation': 0
            }
        }
    
    async def animate_phoneme(self, phoneme_shape: str, duration: float) -> np.ndarray:
        """Generate real-time mouth animation for a phoneme"""
        try:
            if not self.base_face is not None or not self.face_landmarks:
                self.logger.error("Base face not loaded")
                return self.base_face
            
            # Get transformation parameters
            transform = self.phoneme_transforms.get(phoneme_shape, self.phoneme_transforms['mouth_closed'])
            
            # Create animated face
            animated_face = self._apply_mouth_transform(self.base_face.copy(), transform)
            
            return animated_face
            
        except Exception as e:
            self.logger.error(f"Error animating phoneme {phoneme_shape}: {e}")
            return self.base_face
    
    def _apply_mouth_transform(self, face_image: np.ndarray, transform: Dict) -> np.ndarray:
        """Apply real-time mouth transformation to face image"""
        try:
            if not self.face_landmarks or not self.face_dimensions:
                return face_image
            
            h, w = face_image.shape[:2]
            
            # Extract mouth landmark coordinates
            mouth_points = []
            for idx in self.mouth_landmarks:
                if idx < len(self.face_landmarks.landmark):
                    landmark = self.face_landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    mouth_points.append([x, y])
            
            if len(mouth_points) < 4:
                return face_image
            
            mouth_points = np.array(mouth_points, dtype=np.int32)
            
            # Calculate mouth center and bounding box
            mouth_center = np.mean(mouth_points, axis=0).astype(int)
            x_min, y_min = np.min(mouth_points, axis=0)
            x_max, y_max = np.max(mouth_points, axis=0)
            
            # Expand region slightly for better blending
            margin = 20
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)
            
            # Extract mouth region
            mouth_region = face_image[y_min:y_max, x_min:x_max]
            if mouth_region.size == 0:
                return face_image
            
            # Apply transformations
            transformed_mouth = self._transform_mouth_region(
                mouth_region, 
                transform, 
                (mouth_center[0] - x_min, mouth_center[1] - y_min)
            )
            
            # Blend back into face
            result_face = face_image.copy()
            
            # Create smooth blending mask
            mask = self._create_blending_mask(mouth_region.shape[:2])
            
            # Apply transformed mouth with blending
            for c in range(3):  # RGB channels
                result_face[y_min:y_max, x_min:x_max, c] = (
                    mask * transformed_mouth[:, :, c] + 
                    (1 - mask) * result_face[y_min:y_max, x_min:x_max, c]
                )
            
            return result_face.astype(np.uint8)
            
        except Exception as e:
            self.logger.error(f"Error applying mouth transform: {e}")
            return face_image
    
    def _transform_mouth_region(self, mouth_region: np.ndarray, transform: Dict, center: Tuple[int, int]) -> np.ndarray:
        """Apply specific transformations to mouth region"""
        try:
            h, w = mouth_region.shape[:2]
            
            # Create transformation matrix
            # Scale transformations
            scale_x = transform.get('horizontal_scale', 1.0)
            scale_y = transform.get('vertical_scale', 1.0)
            
            # Translation for jaw drop
            jaw_drop = transform.get('jaw_drop', 0)
            
            # Create affine transformation matrix
            M = cv2.getRotationMatrix2D(center, 0, 1.0)
            M[0, 0] *= scale_x  # Horizontal scaling
            M[1, 1] *= scale_y  # Vertical scaling
            M[1, 2] += jaw_drop  # Jaw drop (vertical translation)
            
            # Apply transformation
            transformed = cv2.warpAffine(mouth_region, M, (w, h), flags=cv2.INTER_LANCZOS4)
            
            # Apply additional effects
            if transform.get('pucker', False):
                # Add lip rounding effect
                transformed = self._apply_pucker_effect(transformed, center)
            
            return transformed
            
        except Exception as e:
            self.logger.error(f"Error transforming mouth region: {e}")
            return mouth_region
    
    def _apply_pucker_effect(self, mouth_region: np.ndarray, center: Tuple[int, int]) -> np.ndarray:
        """Apply lip puckering effect for 'OO' sounds"""
        try:
            h, w = mouth_region.shape[:2]
            
            # Create radial compression effect
            y, x = np.ogrid[:h, :w]
            cx, cy = center
            
            # Calculate distance from center
            dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
            
            # Create compression factor (stronger near center)
            max_dist = min(w, h) // 4
            compression = np.clip(1.0 - (dist_from_center / max_dist) * 0.3, 0.7, 1.0)
            
            # Apply radial compression
            result = mouth_region.copy()
            for c in range(3):
                result[:, :, c] = (mouth_region[:, :, c] * compression).astype(np.uint8)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error applying pucker effect: {e}")
            return mouth_region
    
    def _create_blending_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Create smooth blending mask for mouth region"""
        h, w = shape
        
        # Create elliptical mask with soft edges
        y, x = np.ogrid[:h, :w]
        center_x, center_y = w // 2, h // 2
        
        # Elliptical mask
        mask = ((x - center_x) / (w // 2))**2 + ((y - center_y) / (h // 2))**2
        mask = 1.0 - np.clip(mask, 0, 1)
        
        # Smooth the edges
        mask = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 0)
        
        return mask
    
    def cleanup(self):
        """Cleanup resources"""
        if self.face_mesh:
            self.face_mesh.close()
        self.logger.info("Real-time Face Manipulator cleaned up") 