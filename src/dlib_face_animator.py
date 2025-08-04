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
        
        # Enhanced audio analysis parameters
        self.audio_history = []
        self.max_history = 10
        
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
            
            # Log mouth detection
            mouth_center = np.mean(self.mouth_landmarks, axis=0)
            self.logger.info(f"🎬 dlib: Face loaded with 68 landmarks")
            self.logger.info(f"👄 dlib: Mouth landmarks detected at points 48-67")
            self.logger.info(f"📍 dlib: Mouth center: {mouth_center}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading base face: {e}")
            return False
    
    def generate_face_from_audio(self, audio_data: bytes, duration: float) -> np.ndarray:
        """Generate animated face from audio data"""
        try:
            if self.base_face is None:
                self.logger.error("Base face not loaded")
                return None
            
            # Convert audio to numpy array
            audio_array = self._bytes_to_audio_array(audio_data)
            
            # Enhanced audio analysis
            amplitude, frequency, phoneme_type = self._enhanced_audio_analysis(audio_array)
            
            # Debug logging
            self.logger.info(f"🎭 dlib: amp={amplitude:.3f}, freq={frequency:.3f}, phoneme={phoneme_type}")
            
            # Try new seamless approach first
            try:
                animated_face = self._apply_seamless_mouth_deformation(amplitude, frequency, phoneme_type)
                if animated_face is not None and animated_face.shape == self.base_face.shape:
                    self.logger.debug("✅ Seamless deformation successful")
                    return animated_face
                else:
                    self.logger.warning("⚠️ Seamless deformation failed, trying fallback")
            except Exception as e:
                self.logger.warning(f"⚠️ Seamless deformation error: {e}, trying fallback")
            
            # Fallback to simple deformation if seamless fails
            try:
                self.logger.info("🔄 Using fallback mouth deformation")
                animated_face = self._fallback_simple_deformation(amplitude, frequency)
                return animated_face
            except Exception as e:
                self.logger.error(f"❌ Fallback deformation also failed: {e}")
                return self.base_face
            
        except Exception as e:
            self.logger.error(f"Error generating face from audio: {e}")
            return self.base_face
    
    def generate_face_for_audio_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Generate face for audio chunk - bridge method for face_animator.py"""
        try:
            if self.base_face is None:
                return self.base_face
            
            # Enhanced audio analysis - use audio_chunk directly (it's already a numpy array)
            amplitude, frequency, phoneme_type = self._enhanced_audio_analysis(audio_chunk)
            
            # Debug logging - use print for immediate visibility
            print(f"🎭 DLIB DEBUG: amp={amplitude:.3f}, freq={frequency:.3f}, phoneme={phoneme_type}")
            self.logger.info(f"🎭 dlib: amp={amplitude:.3f}, freq={frequency:.3f}, phoneme={phoneme_type}")
            
            # Try new seamless approach first
            try:
                animated_face = self._apply_seamless_mouth_deformation(amplitude, frequency, phoneme_type)
                if animated_face is not None and animated_face.shape == self.base_face.shape:
                    self.logger.debug("✅ Seamless deformation successful")
                    return animated_face
                else:
                    self.logger.warning("⚠️ Seamless deformation failed, trying fallback")
            except Exception as e:
                self.logger.warning(f"⚠️ Seamless deformation error: {e}, trying fallback")
            
            # Fallback to simple deformation if seamless fails
            try:
                self.logger.info("🔄 Using fallback mouth deformation")
                animated_face = self._fallback_simple_deformation(amplitude, frequency)
                return animated_face
            except Exception as e:
                self.logger.error(f"❌ Fallback deformation also failed: {e}")
                return self.base_face
            
        except Exception as e:
            self.logger.error(f"Error in generate_face_for_audio_chunk: {e}")
            return self.base_face
    
    def _enhanced_audio_analysis(self, audio_array: np.ndarray) -> Tuple[float, float, str]:
        """Enhanced audio analysis for better lip-sync"""
        try:
            print(f"🔍 ENHANCED AUDIO ANALYSIS: Called with {len(audio_array)} samples")
            # Basic amplitude and frequency analysis
            amplitude = self._analyze_amplitude(audio_array)
            frequency = self._analyze_frequency(audio_array)
            
            # Add to history for trend analysis
            self.audio_history.append((amplitude, frequency))
            if len(self.audio_history) > self.max_history:
                self.audio_history.pop(0)
            
            # Analyze trends for better lip-sync
            if len(self.audio_history) >= 3:
                recent_amps = [a for a, f in self.audio_history[-3:]]
                recent_freqs = [f for a, f in self.audio_history[-3:]]
                
                # Detect phoneme types based on audio patterns
                avg_amp = np.mean(recent_amps)
                avg_freq = np.mean(recent_freqs)
                amp_variance = np.var(recent_amps)
                
                # Phoneme classification for better mouth shapes
                if avg_freq > 0.6 and amp_variance > 0.1:
                    phoneme_type = "vowel"  # Wide open mouth for vowels
                elif avg_amp > 0.7:
                    phoneme_type = "consonant"  # Moderate opening for consonants
                elif avg_freq < 0.3:
                    phoneme_type = "closed"  # Nearly closed for quiet sounds
                else:
                    phoneme_type = "neutral"
            else:
                phoneme_type = "neutral"
            
            return amplitude, frequency, phoneme_type
            
        except Exception as e:
            self.logger.error(f"Error in enhanced audio analysis: {e}")
            return 0.5, 0.5, "neutral"
    
    def _apply_seamless_mouth_deformation(self, amplitude: float, frequency: float, phoneme_type: str) -> np.ndarray:
        """Apply seamless mouth deformation without visible boxes"""
        try:
            # Create a copy of the base face
            result = self.base_face.copy()
            
            # Get mouth landmarks
            mouth_points = self.original_mouth_points.copy().astype(np.float32)
            mouth_center = np.mean(mouth_points, axis=0)
            
            # Calculate mouth dimensions
            mouth_width = np.max(mouth_points[:, 0]) - np.min(mouth_points[:, 0])
            mouth_height = np.max(mouth_points[:, 1]) - np.min(mouth_points[:, 1])
            
            # Apply phoneme-specific deformations - INCREASED INTENSITY
            if phoneme_type == "vowel":
                # Wide open mouth for vowels (A, E, I, O, U)
                jaw_drop = amplitude * 120  # Very dramatic jaw drop (increased from 80)
                width_stretch = 1.0 + (frequency * 1.2)  # Wide stretch (increased from 0.6)
                height_stretch = 1.0 + (amplitude * 1.2)  # Tall opening (increased from 0.8)
                
            elif phoneme_type == "consonant":
                # Moderate opening for consonants (B, P, M, etc.)
                jaw_drop = amplitude * 80  # Moderate jaw drop (increased from 40)
                width_stretch = 0.8 + (frequency * 0.8)  # Slight stretch (increased from 0.3)
                height_stretch = 0.7 + (amplitude * 0.8)  # Moderate height (increased from 0.4)
                
            elif phoneme_type == "closed":
                # Nearly closed for quiet sounds
                jaw_drop = amplitude * 30  # Minimal jaw drop (increased from 15)
                width_stretch = 0.7 + (frequency * 0.4)  # Slight compression (increased from 0.2)
                height_stretch = 0.6 + (amplitude * 0.4)  # Minimal height (increased from 0.2)
                
            else:  # neutral
                jaw_drop = amplitude * 60  # Increased from 30
                width_stretch = 0.8 + (frequency * 0.8)  # Increased from 0.4
                height_stretch = 0.7 + (amplitude * 0.8)  # Increased from 0.5
            
            # Apply deformations with seamless blending
            new_mouth_points = self._calculate_deformed_mouth_points(
                mouth_points, mouth_center, jaw_drop, width_stretch, height_stretch
            )
            
            # Apply seamless warping without visible boxes
            result = self._seamless_mouth_warp(result, mouth_points, new_mouth_points)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in seamless mouth deformation: {e}")
            return self.base_face
    
    def _calculate_deformed_mouth_points(self, points: np.ndarray, center: np.ndarray, 
                                       jaw_drop: float, width_stretch: float, height_stretch: float) -> np.ndarray:
        """Calculate new mouth point positions with natural deformation"""
        try:
            new_points = points.copy()
            
            # Define lip regions for more natural movement
            upper_lip_indices = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]  # Upper lip
            lower_lip_indices = [60, 61, 62, 63, 64, 65, 66, 67]  # Lower lip
            corner_indices = [48, 54]  # Mouth corners
            
            # Apply jaw drop to lower lip
            for i in lower_lip_indices:
                if i < len(new_points):
                    # More drop for points closer to center
                    distance_from_center = abs(new_points[i][0] - center[0]) / (np.max(points[:, 0]) - np.min(points[:, 0]))
                    drop_factor = 1.0 - distance_from_center * 0.5  # Less drop at corners
                    new_points[i][1] += jaw_drop * drop_factor
            
            # Apply width stretching
            for i, point in enumerate(new_points):
                # Stretch horizontally from center
                dx = (point[0] - center[0]) * (width_stretch - 1.0)
                new_points[i][0] += dx
            
            # Apply height stretching
            for i, point in enumerate(new_points):
                # Stretch vertically from center
                dy = (point[1] - center[1]) * (height_stretch - 1.0)
                new_points[i][1] += dy
            
            return new_points
            
        except Exception as e:
            self.logger.error(f"Error calculating deformed mouth points: {e}")
            return points
    
    def _seamless_mouth_warp(self, face: np.ndarray, src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
        """Apply seamless mouth warping without visible boxes"""
        try:
            # Create a copy of the face
            result = face.copy()
            
            # Calculate mouth region with natural boundaries
            mouth_bbox = cv2.boundingRect(src_points.astype(np.int32))
            x, y, w, h = mouth_bbox
            
            # Expand region to include surrounding face area for seamless blending
            expansion = 40  # Larger area for better blending
            x = max(0, x - expansion)
            y = max(0, y - expansion)
            w = min(face.shape[1] - x, w + 2 * expansion)
            h = min(face.shape[0] - y, h + 2 * expansion)
            
            # Extract the region
            region = face[y:y+h, x:x+w]
            
            # Adjust point coordinates to region
            src_region = src_points - [x, y]
            dst_region = dst_points - [x, y]
            
            # Create a mask for the mouth area
            mask = np.zeros((h, w), dtype=np.uint8)
            mouth_contour = src_region.astype(np.int32)
            cv2.fillPoly(mask, [mouth_contour], 255)
            
            # Apply Gaussian blur to create soft edges
            mask = cv2.GaussianBlur(mask, (21, 21), 0)
            
            # Normalize mask
            mask = mask.astype(np.float32) / 255.0
            mask = np.stack([mask, mask, mask], axis=2)
            
            # Apply perspective transform to the region
            try:
                # Calculate homography matrix
                H = cv2.findHomography(src_region, dst_region, cv2.RANSAC, 5.0)[0]
                
                # Apply transformation
                warped_region = cv2.warpPerspective(region, H, (w, h), 
                                                  flags=cv2.INTER_LINEAR, 
                                                  borderMode=cv2.BORDER_REFLECT)
                
                # Blend using the mask for seamless integration
                blended_region = region * (1 - mask) + warped_region * mask
                
                # Apply to result
                result[y:y+h, x:x+w] = blended_region
                
            except Exception as e:
                # Fallback to simple scaling if homography fails
                self.logger.debug(f"Homography failed, using fallback: {e}")
                result = self._fallback_mouth_deformation(result, src_points, dst_points)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in seamless mouth warp: {e}")
            return face
    
    def _fallback_mouth_deformation(self, face: np.ndarray, src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
        """Fallback mouth deformation using simple scaling"""
        try:
            result = face.copy()
            
            # Calculate center and scale
            src_center = np.mean(src_points, axis=0)
            dst_center = np.mean(dst_points, axis=0)
            
            # Calculate scale factors
            src_scale = np.mean(np.linalg.norm(src_points - src_center, axis=1))
            dst_scale = np.mean(np.linalg.norm(dst_points - dst_center, axis=1))
            
            if src_scale > 0:
                scale_factor = dst_scale / src_scale
                scale_factor = np.clip(scale_factor, 0.7, 1.5)
                
                # Apply scaling around mouth center
                h, w = face.shape[:2]
                center = (int(src_center[0]), int(src_center[1]))
                
                # Create scaling matrix
                M = cv2.getRotationMatrix2D(center, 0, scale_factor)
                
                # Apply transformation
                warped = cv2.warpAffine(face, M, (w, h), 
                                       flags=cv2.INTER_LINEAR, 
                                       borderMode=cv2.BORDER_REFLECT)
                
                # Blend with original
                alpha = 0.6
                result = cv2.addWeighted(face, 1-alpha, warped, alpha, 0)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in fallback deformation: {e}")
            return face
    
    def _bytes_to_audio_array(self, audio_data: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array"""
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            # Normalize to float
            audio_array = audio_array.astype(np.float32) / 32768.0
            return audio_array
        except Exception as e:
            self.logger.error(f"Error converting audio bytes: {e}")
            return np.zeros(1024, dtype=np.float32)
    
    def _analyze_amplitude(self, audio_array: np.ndarray) -> float:
        """Analyze audio amplitude"""
        try:
            if len(audio_array) == 0:
                return 0.0
            
            # Calculate RMS amplitude
            rms = np.sqrt(np.mean(audio_array**2))
            
            # Apply non-linear scaling for better sensitivity
            amplitude = np.tanh(rms * 4.0)  # Enhanced sensitivity
            
            print(f"📊 AMPLITUDE ANALYSIS: rms={rms:.4f}, amplitude={amplitude:.4f}")
            return float(amplitude)
        except Exception as e:
            self.logger.error(f"Error analyzing amplitude: {e}")
            return 0.5
    
    def _analyze_frequency(self, audio_array: np.ndarray) -> float:
        """Analyze audio frequency content"""
        try:
            if len(audio_array) < 64:
                return 0.5
            
            # Calculate FFT for frequency analysis
            fft = np.fft.fft(audio_array)
            fft_magnitude = np.abs(fft[:len(fft)//2])
            
            # Calculate dominant frequency
            freqs = np.fft.fftfreq(len(audio_array))[:len(fft)//2]
            dominant_freq_idx = np.argmax(fft_magnitude)
            dominant_freq = abs(freqs[dominant_freq_idx])
            
            # Normalize frequency (0-1 range) - use a much lower base frequency for better sensitivity
            frequency = min(dominant_freq / 50.0, 1.0)  # Normalize to 50Hz instead of 1kHz for better sensitivity
            
            print(f"🎵 FREQUENCY ANALYSIS: dominant_freq={dominant_freq:.2f}Hz, normalized={frequency:.4f}")
            return float(frequency)
        except Exception as e:
            self.logger.error(f"Error analyzing frequency: {e}")
            return 0.5 

    def _fallback_simple_deformation(self, amplitude: float, frequency: float) -> np.ndarray:
        """Simple fallback deformation that we know works"""
        try:
            # Create a copy of original mouth landmarks
            new_mouth_points = self.original_mouth_points.copy().astype(np.float32)
            
            # Calculate mouth center
            mouth_center = np.mean(new_mouth_points, axis=0)
            
            # Apply simple deformations based on audio
            
            # 1. Jaw drop (move bottom lip down) - INCREASED INTENSITY
            jaw_drop = amplitude * 100  # Dramatic movement (increased from 60)
            bottom_lip_indices = [60, 61, 62, 63, 64, 65, 66, 67]  # Bottom lip in mouth landmarks
            for i in bottom_lip_indices:
                if i < len(new_mouth_points):
                    new_mouth_points[i][1] += jaw_drop
            
            # 2. Lip width (squeeze/stretch horizontally) - INCREASED INTENSITY
            width_factor = 0.6 + (frequency * 1.2)  # 0.6 to 1.8 range (increased from 0.7-1.5)
            for i, point in enumerate(new_mouth_points):
                # Move points horizontally relative to center
                dx = (point[0] - mouth_center[0]) * (width_factor - 1.0)
                new_mouth_points[i][0] += dx
            
            # 3. Lip height (vertical scaling) - INCREASED INTENSITY
            height_factor = 0.6 + (amplitude * 1.0)  # 0.6 to 1.6 range (increased from 0.8-1.4)
            for i, point in enumerate(new_mouth_points):
                # Move points vertically relative to center
                dy = (point[1] - mouth_center[1]) * (height_factor - 1.0)
                new_mouth_points[i][1] += dy
            
            # Apply simple scaling transformation
            result_face = self._simple_mouth_warp(self.base_face.copy(), self.original_mouth_points, new_mouth_points)
            
            return result_face
            
        except Exception as e:
            self.logger.error(f"Error in fallback deformation: {e}")
            return self.base_face
    
    def _simple_mouth_warp(self, face: np.ndarray, src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
        """Simple mouth warping that we know works"""
        try:
            result = face.copy()
            
            # Calculate center and scale
            src_center = np.mean(src_points, axis=0)
            dst_center = np.mean(dst_points, axis=0)
            
            # Calculate scale factors
            src_scale = np.mean(np.linalg.norm(src_points - src_center, axis=1))
            dst_scale = np.mean(np.linalg.norm(dst_points - dst_center, axis=1))
            
            if src_scale > 0:
                scale_factor = dst_scale / src_scale
                scale_factor = np.clip(scale_factor, 0.6, 1.8)
                
                # Apply scaling around mouth center
                h, w = face.shape[:2]
                center = (int(src_center[0]), int(src_center[1]))
                
                # Create scaling matrix
                M = cv2.getRotationMatrix2D(center, 0, scale_factor)
                
                # Apply transformation
                warped = cv2.warpAffine(face, M, (w, h), 
                                       flags=cv2.INTER_LINEAR, 
                                       borderMode=cv2.BORDER_REFLECT)
                
                # Blend with original
                alpha = 0.8
                result = cv2.addWeighted(face, 1-alpha, warped, alpha, 0)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in simple mouth warp: {e}")
            return face 