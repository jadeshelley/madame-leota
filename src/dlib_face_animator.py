"""
dlib-based Facial Landmark Animation for Madame Leota
Uses real facial landmark detection for precise mouth manipulation
"""

import cv2
import numpy as np
import dlib
import imutils
from imutils import face_utils
import logging
import asyncio
from typing import Tuple, List, Optional
from config import *


class DlibFaceAnimator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize dlib face detector and predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = None
        self.predictor_path = "shape_predictor_68_face_landmarks.dat"
        
        # Face data
        self.base_face = None
        self.base_landmarks = None
        self.mouth_landmarks = None  # Indices 48-67 are mouth landmarks
        
        # Original mouth coordinates for reference
        self.original_mouth_points = None
        
        self._initialize_predictor()
        
    def _initialize_predictor(self):
        """Initialize the facial landmark predictor"""
        try:
            # Try to load the predictor
            self.predictor = dlib.shape_predictor(self.predictor_path)
            self.logger.info("✅ dlib facial landmark predictor loaded successfully")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not load dlib predictor: {e}")
            self.logger.info("📥 Downloading facial landmark predictor...")
            self._download_predictor()
            
    def _download_predictor(self):
        """Download the facial landmark predictor if not found"""
        try:
            import urllib.request
            import bz2
            import os
            
            url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            compressed_file = "shape_predictor_68_face_landmarks.dat.bz2"
            
            print("🔽 Downloading facial landmark predictor...")
            urllib.request.urlretrieve(url, compressed_file)
            
            print("📂 Extracting predictor...")
            with bz2.BZ2File(compressed_file, 'rb') as f_in:
                with open(self.predictor_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            
            # Clean up compressed file
            os.remove(compressed_file)
            
            # Load the predictor
            self.predictor = dlib.shape_predictor(self.predictor_path)
            print("✅ Facial landmark predictor ready!")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to download predictor: {e}")
            raise
    
    def load_base_face(self, face_image_path: str) -> bool:
        """Load base face and detect landmarks"""
        try:
            self.base_face = cv2.imread(face_image_path)
            if self.base_face is None:
                self.logger.error(f"Could not load face image: {face_image_path}")
                return False
            
            # Convert to grayscale for landmark detection
            gray = cv2.cvtColor(self.base_face, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.detector(gray)
            if len(faces) == 0:
                self.logger.error("No faces detected in base image")
                return False
            
            # Use the largest face
            face = max(faces, key=lambda rect: rect.width() * rect.height())
            
            # Get facial landmarks
            landmarks = self.predictor(gray, face)
            self.base_landmarks = face_utils.shape_to_np(landmarks)
            
            # Extract mouth landmarks (points 48-67)
            self.mouth_landmarks = self.base_landmarks[48:68]
            self.original_mouth_points = self.mouth_landmarks.copy()
            
            print(f"✅ dlib: Face loaded with {len(self.base_landmarks)} landmarks")
            print(f"👄 dlib: Mouth landmarks detected at points 48-67")
            print(f"📍 dlib: Mouth center: {np.mean(self.mouth_landmarks, axis=0)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading base face: {e}")
            return False
    
    def generate_face_from_audio(self, audio_data: bytes, duration: float) -> np.ndarray:
        """Generate face with mouth movement based on audio"""
        try:
            if self.base_face is None or self.mouth_landmarks is None:
                return self.base_face
            
            # Analyze audio for mouth parameters
            audio_array = self._bytes_to_audio_array(audio_data)
            if len(audio_array) == 0:
                return self.base_face
            
            # Calculate mouth movement parameters
            amplitude = self._analyze_amplitude(audio_array)
            frequency = self._analyze_frequency(audio_array)
            
            # Generate mouth deformation based on audio
            deformed_face = self._apply_mouth_deformation(
                self.base_face.copy(), amplitude, frequency
            )
            
            print(f"🎭 dlib: amp={amplitude:.3f}, freq={frequency:.3f}")
            
            return deformed_face
            
        except Exception as e:
            self.logger.error(f"Error generating face: {e}")
            return self.base_face if self.base_face is not None else np.zeros((512, 512, 3), dtype=np.uint8)
    
    def _apply_mouth_deformation(self, face: np.ndarray, amplitude: float, frequency: float) -> np.ndarray:
        """Apply mouth deformation using facial landmarks"""
        try:
            # Create a copy of original mouth landmarks
            new_mouth_points = self.original_mouth_points.copy().astype(np.float32)
            
            # Calculate mouth center
            mouth_center = np.mean(new_mouth_points, axis=0)
            
            # Apply deformations based on audio
            
            # 1. Jaw drop (move bottom lip down)
            jaw_drop = amplitude * 30  # More dramatic movement
            bottom_lip_indices = [15, 16, 17, 18, 19]  # Bottom lip in mouth landmarks
            for i in bottom_lip_indices:
                if i < len(new_mouth_points):
                    new_mouth_points[i][1] += jaw_drop
            
            # 2. Lip width (squeeze/stretch horizontally)
            width_factor = 0.8 + (frequency * 0.4)  # 0.8 to 1.2 range
            for i, point in enumerate(new_mouth_points):
                # Move points horizontally relative to center
                dx = (point[0] - mouth_center[0]) * (width_factor - 1.0)
                new_mouth_points[i][0] += dx
            
            # 3. Lip height (vertical scaling)
            height_factor = 0.9 + (amplitude * 0.3)  # 0.9 to 1.2 range
            for i, point in enumerate(new_mouth_points):
                # Move points vertically relative to center
                dy = (point[1] - mouth_center[1]) * (height_factor - 1.0)
                new_mouth_points[i][1] += dy
            
            # Apply the deformation using perspective transform
            result_face = self._warp_mouth_region(face, self.original_mouth_points, new_mouth_points)
            
            return result_face
            
        except Exception as e:
            self.logger.error(f"Error in mouth deformation: {e}")
            return face
    
    def _warp_mouth_region(self, face: np.ndarray, src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
        """Warp the mouth region using triangulation"""
        try:
            # Create a copy of the face
            result = face.copy()
            
            # Get bounding rectangle of mouth region
            x, y, w, h = cv2.boundingRect(src_points.astype(np.int32))
            
            # Add padding around mouth
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(face.shape[1] - x, w + 2 * padding)
            h = min(face.shape[0] - y, h + 2 * padding)
            
            # Extract mouth region
            mouth_region = face[y:y+h, x:x+w]
            
            # Adjust point coordinates to mouth region
            src_region = src_points - [x, y]
            dst_region = dst_points - [x, y]
            
            # Create triangulation for the mouth region
            rect = (0, 0, w, h)
            subdiv = cv2.Subdiv2D(rect)
            
            # Add points to subdivision
            for point in src_region:
                if 0 <= point[0] < w and 0 <= point[1] < h:
                    subdiv.insert((int(point[0]), int(point[1])))
            
            # Get triangles
            triangles = subdiv.getTriangleList()
            
            if len(triangles) > 0:
                # Apply triangular warping
                warped_region = self._apply_triangular_warp(mouth_region, src_region, dst_region, triangles)
                
                # Blend back into original face
                result[y:y+h, x:x+w] = warped_region
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in mouth warping: {e}")
            return face
    
    def _apply_triangular_warp(self, img: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray, triangles: np.ndarray) -> np.ndarray:
        """Apply triangular warping to image"""
        try:
            warped = img.copy()
            
            for triangle in triangles:
                # Get triangle vertices
                t = triangle.reshape(3, 2)
                
                # Find corresponding points in src and dst
                src_tri = []
                dst_tri = []
                
                for vertex in t:
                    # Find closest point in src_pts
                    distances = np.linalg.norm(src_pts - vertex, axis=1)
                    closest_idx = np.argmin(distances)
                    
                    if distances[closest_idx] < 10:  # Close enough
                        src_tri.append(src_pts[closest_idx])
                        dst_tri.append(dst_pts[closest_idx])
                
                if len(src_tri) == 3:
                    # Apply affine transformation for this triangle
                    src_tri = np.float32(src_tri)
                    dst_tri = np.float32(dst_tri)
                    
                    # Get transformation matrix
                    warp_mat = cv2.getAffineTransform(src_tri, dst_tri)
                    
                    # Apply transformation to triangle region
                    warped = cv2.warpAffine(warped, warp_mat, (img.shape[1], img.shape[0]))
            
            return warped
            
        except Exception as e:
            self.logger.error(f"Error in triangular warp: {e}")
            return img
    
    def _bytes_to_audio_array(self, audio_data: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array"""
        try:
            # Try to interpret as int16 audio data
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767.0
            return audio_array
        except Exception as e:
            self.logger.error(f"Audio conversion error: {e}")
            return np.array([])
    
    def _analyze_amplitude(self, audio_array: np.ndarray) -> float:
        """Analyze audio amplitude"""
        if len(audio_array) == 0:
            return 0.0
        
        rms = np.sqrt(np.mean(audio_array ** 2))
        return np.clip(rms * 2.0, 0.0, 1.0)  # Scale and clip
    
    def _analyze_frequency(self, audio_array: np.ndarray) -> float:
        """Simple frequency analysis"""
        if len(audio_array) < 2:
            return 0.5
        
        # Simple zero-crossing rate as frequency indicator
        zero_crossings = np.sum(np.diff(np.sign(audio_array)) != 0)
        frequency_indicator = min(zero_crossings / len(audio_array), 1.0)
        
        return frequency_indicator 