"""
Real-Time Face Manipulator for Madame Leota
Pi-optimized version using dlib for facial landmarks
Much more compatible with Raspberry Pi ARM architecture
"""

import cv2
import numpy as np
import dlib
import logging
import asyncio
import time
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from config import *

class RealtimeFaceManipulator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize dlib face detector and landmark predictor
        try:
            self.detector = dlib.get_frontal_face_detector()
            
            # Try to load shape predictor (download if needed)
            predictor_path = self._get_shape_predictor()
            self.predictor = dlib.shape_predictor(predictor_path)
            
            self.logger.info("Dlib face detector initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize dlib: {e}")
            raise
        
        # Base face image (the realistic face)
        self.base_face = None
        self.face_landmarks = None
        self.face_dimensions = None
        
        # Mouth landmark indices (dlib 68-point model)
        self.mouth_landmarks = list(range(48, 68))  # Points 48-67 are mouth region
        
        # Phoneme to mouth transformation mapping
        self.phoneme_transforms = self._create_phoneme_transforms()
        
        self.logger.info("Real-time Face Manipulator (dlib) initialized")
    
    def _get_shape_predictor(self) -> str:
        """Get or download the dlib shape predictor model"""
        predictor_path = Path("models/shape_predictor_68_face_landmarks.dat")
        
        if not predictor_path.exists():
            # Create models directory
            predictor_path.parent.mkdir(exist_ok=True)
            
            self.logger.info("Downloading dlib shape predictor model...")
            import urllib.request
            import bz2
            
            # Download compressed model
            url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            compressed_path = predictor_path.with_suffix('.dat.bz2')
            
            try:
                urllib.request.urlretrieve(url, compressed_path)
                
                # Decompress
                with bz2.BZ2File(compressed_path, 'rb') as f_in:
                    with open(predictor_path, 'wb') as f_out:
                        f_out.write(f_in.read())
                
                # Remove compressed file
                compressed_path.unlink()
                
                self.logger.info("Shape predictor model downloaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to download shape predictor: {e}")
                # Create a simple fallback
                self._create_simple_predictor(predictor_path)
        
        return str(predictor_path)
    
    def _create_simple_predictor(self, path: Path):
        """Create a simple fallback if download fails"""
        self.logger.warning("Using simplified landmark detection")
        # Create empty file as placeholder
        path.touch()
    
    def load_base_face(self, face_image_path: str) -> bool:
        """Load the base face image for manipulation"""
        try:
            # Load the main face image (mouth_closed.png)
            self.base_face = cv2.imread(face_image_path)
            if self.base_face is None:
                self.logger.error(f"Could not load base face: {face_image_path}")
                return False
            
            # Convert to grayscale for dlib
            gray = cv2.cvtColor(self.base_face, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.detector(gray)
            
            if len(faces) > 0:
                # Get landmarks for the first face
                face = faces[0]
                landmarks = self.predictor(gray, face)
                
                # Convert to numpy array
                self.face_landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
                h, w = self.base_face.shape[:2]
                self.face_dimensions = (w, h)
                
                self.logger.info("Base face loaded and landmarks detected with dlib")
                return True
            else:
                self.logger.error("No face detected in base image")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading base face: {e}")
            return False
    
    def _create_phoneme_transforms(self) -> Dict:
        """Create mouth transformation parameters for different phonemes"""
        return {
            # Vowels - mouth opening variations
            'mouth_open': {
                'vertical_scale': 1.6,    # Stretch mouth vertically
                'horizontal_scale': 1.0,
                'jaw_drop': 12,           # Lower jaw position
                'intensity': 0.8
            },
            'mouth_wide': {
                'vertical_scale': 0.9,
                'horizontal_scale': 1.4,  # Stretch mouth horizontally  
                'jaw_drop': 4,
                'intensity': 0.7
            },
            'mouth_round': {
                'vertical_scale': 1.2,
                'horizontal_scale': 0.9,  # Compress horizontally
                'jaw_drop': 6,
                'pucker': True,           # Round the lips
                'intensity': 0.6
            },
            'mouth_narrow': {
                'vertical_scale': 1.05,
                'horizontal_scale': 0.95,
                'jaw_drop': 2,
                'intensity': 0.4
            },
            'mouth_closed': {
                'vertical_scale': 1.0,
                'horizontal_scale': 1.0,
                'jaw_drop': 0,
                'intensity': 0.0
            }
        }
    
    async def animate_phoneme(self, phoneme_shape: str, duration: float) -> np.ndarray:
        """Generate real-time mouth animation for a phoneme"""
        try:
            if self.base_face is None or self.face_landmarks is None:
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
            if self.face_landmarks is None:
                return face_image
            
            h, w = face_image.shape[:2]
            
            # Get mouth landmarks (points 48-67 in dlib 68-point model)
            mouth_points = self.face_landmarks[self.mouth_landmarks]
            
            if len(mouth_points) < 4:
                return face_image
            
            # Calculate mouth center and bounding box
            mouth_center = np.mean(mouth_points, axis=0).astype(int)
            x_min, y_min = np.min(mouth_points, axis=0)
            x_max, y_max = np.max(mouth_points, axis=0)
            
            # Expand region for better blending
            margin = 15
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)
            
            # Extract mouth region
            mouth_region = face_image[y_min:y_max, x_min:x_max]
            if mouth_region.size == 0:
                return face_image
            
            # Apply lightweight transformations
            transformed_mouth = self._transform_mouth_region_simple(
                mouth_region, 
                transform, 
                (mouth_center[0] - x_min, mouth_center[1] - y_min)
            )
            
            # Blend back into face
            result_face = face_image.copy()
            
            # Simple blending with reduced intensity for Pi performance
            intensity = transform.get('intensity', 0.5)
            blend_region = result_face[y_min:y_max, x_min:x_max]
            
            # Weighted blend
            result_face[y_min:y_max, x_min:x_max] = (
                intensity * transformed_mouth + 
                (1 - intensity) * blend_region
            ).astype(np.uint8)
            
            return result_face
            
        except Exception as e:
            self.logger.error(f"Error applying mouth transform: {e}")
            return face_image
    
    def _transform_mouth_region_simple(self, mouth_region: np.ndarray, transform: Dict, center: Tuple[int, int]) -> np.ndarray:
        """Apply lightweight transformations optimized for Pi"""
        try:
            h, w = mouth_region.shape[:2]
            
            # Get transformation parameters
            scale_x = transform.get('horizontal_scale', 1.0)
            scale_y = transform.get('vertical_scale', 1.0)
            jaw_drop = transform.get('jaw_drop', 0)
            
            # Create simple affine transformation
            cx, cy = center
            
            # Build transformation matrix
            M = np.array([
                [scale_x, 0, cx * (1 - scale_x)],
                [0, scale_y, cy * (1 - scale_y) + jaw_drop]
            ], dtype=np.float32)
            
            # Apply transformation
            transformed = cv2.warpAffine(
                mouth_region, M, (w, h), 
                flags=cv2.INTER_LINEAR,  # Faster than LANCZOS for Pi
                borderMode=cv2.BORDER_REFLECT
            )
            
            # Apply pucker effect if needed (simplified)
            if transform.get('pucker', False):
                transformed = self._apply_simple_pucker(transformed, center)
            
            return transformed
            
        except Exception as e:
            self.logger.error(f"Error in simple mouth transform: {e}")
            return mouth_region
    
    def _apply_simple_pucker(self, mouth_region: np.ndarray, center: Tuple[int, int]) -> np.ndarray:
        """Simple pucker effect for Pi"""
        try:
            # Just apply slight compression - much faster than complex radial effects
            h, w = mouth_region.shape[:2]
            cx, cy = center
            
            # Simple oval mask compression
            M = cv2.getRotationMatrix2D((cx, cy), 0, 0.95)
            result = cv2.warpAffine(mouth_region, M, (w, h))
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in simple pucker: {e}")
            return mouth_region
    
    def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Real-time Face Manipulator (dlib) cleaned up") 