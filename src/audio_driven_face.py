"""
Audio-Driven Face Manipulation for Madame Leota
Creates deepfake-like lip sync by analyzing audio waveform in real-time
Pi-compatible version using only numpy and basic libraries
"""

import cv2
import numpy as np
import logging
import asyncio
import time
import wave
import io
from typing import List, Dict, Tuple, Optional
from config import *

class AudioDrivenFace:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Base face image
        self.base_face = None
        self.face_height, self.face_width = 0, 0
        
        # Audio analysis parameters
        self.sample_rate = SAMPLE_RATE
        self.frame_size = 1024
        
        # Mouth manipulation parameters
        self.mouth_center = None
        
        # Audio-to-visual mapping with history for smoothing
        self.amplitude_history = []
        self.frequency_history = []
        self.smoothing_window = 3  # Reduced for Pi performance
        
        # Movement parameters (tuned for deepfake-like appearance)
        self.jaw_range = (0, int(20 * JAW_SENSITIVITY))
        self.lip_width_range = (0.85, 1.3 * LIP_SENSITIVITY)
        self.lip_height_range = (0.7, 1.6 * JAW_SENSITIVITY)
        
        self.logger.info("Audio-Driven Deepfake Face system initialized")
    
    def load_base_face(self, face_image_path: str) -> bool:
        """Load the base face for manipulation"""
        try:
            self.base_face = cv2.imread(face_image_path)
            if self.base_face is None:
                return False
            
            self.face_height, self.face_width = self.base_face.shape[:2]
            
            # Estimate mouth center (can be made more sophisticated)
            self.mouth_center = (self.face_width // 2, int(self.face_height * 0.72))
            
            self.logger.info("Base face loaded for deepfake-like manipulation")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading base face: {e}")
            return False
    
    async def generate_face_from_audio(self, audio_data: bytes, duration: float) -> np.ndarray:
        """Generate deepfake-like face animation from audio data"""
        try:
            self.logger.info(f"generate_face_from_audio called with {len(audio_data)} bytes, duration {duration}")
            
            if self.base_face is None:
                self.logger.error("Base face is None!")
                return self.base_face
            
            # Convert audio bytes to numpy array
            audio_array = self._bytes_to_audio_array(audio_data)
            self.logger.info(f"Converted to audio array: {len(audio_array)} samples")
            
            if len(audio_array) == 0:
                self.logger.warning("Audio array is empty")
                return self.base_face
            
            # Analyze audio features (simplified for Pi)
            amplitude = self._analyze_amplitude_simple(audio_array)
            dominant_freq = self._analyze_frequency_simple(audio_array)
            speech_energy = self._analyze_speech_energy(audio_array)
            
            self.logger.info(f"Audio analysis - amplitude: {amplitude:.3f}, freq: {dominant_freq:.3f}, energy: {speech_energy:.3f}")
            
            # Smooth features over time for natural movement
            smoothed_amplitude = self._smooth_feature(amplitude, self.amplitude_history)
            smoothed_frequency = self._smooth_feature(dominant_freq, self.frequency_history)
            
            self.logger.info(f"Smoothed - amplitude: {smoothed_amplitude:.3f}, freq: {smoothed_frequency:.3f}")
            
            # Map audio features to facial parameters
            jaw_drop = self._map_amplitude_to_jaw(smoothed_amplitude)
            lip_width = self._map_frequency_to_lip_width(smoothed_frequency)
            lip_height = self._map_amplitude_to_lip_height(smoothed_amplitude)
            micro_movement = self._generate_micro_movement(speech_energy)
            
            self.logger.info(f"Face params - jaw: {jaw_drop:.1f}, width: {lip_width:.3f}, height: {lip_height:.3f}, movement: {micro_movement}")
            
            # Generate deepfake-like face with these parameters
            deepfake_face = self._apply_deepfake_deformation(
                self.base_face.copy(),
                jaw_drop, lip_width, lip_height, micro_movement
            )
            
            self.logger.info(f"Generated deepfake face with shape: {deepfake_face.shape}")
            return deepfake_face
            
        except Exception as e:
            self.logger.error(f"Error in deepfake face generation: {e}")
            self.logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
            return self.base_face
    
    def _bytes_to_audio_array(self, audio_data: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array (Pi-compatible)"""
        try:
            self.logger.info(f"Audio data starts with: {audio_data[:10]}")
            
            # Try different audio format detection
            if audio_data.startswith(b'RIFF'):
                # WAV format
                self.logger.info("Detected WAV format")
                return self._convert_wav_to_array(audio_data)
            elif audio_data.startswith(b'\xff\xfb') or audio_data.startswith(b'\xff\xf3'):
                # MP3 format
                self.logger.info("Detected MP3 format - converting")
                return self._convert_mp3_to_array(audio_data)
            else:
                # Try to detect other formats or raw audio
                self.logger.info("Unknown format, trying raw audio conversion")
                return self._convert_raw_to_array(audio_data)
                
        except Exception as e:
            self.logger.error(f"Audio conversion error: {e}")
            return np.array([])
    
    def _convert_wav_to_array(self, audio_data: bytes) -> np.ndarray:
        """Convert WAV bytes to numpy array"""
        try:
            audio_buffer = io.BytesIO(audio_data)
            with wave.open(audio_buffer, 'rb') as wav_file:
                frames = wav_file.readframes(-1)
                audio_array = np.frombuffer(frames, dtype=np.int16)
                return audio_array.astype(np.float32) / 32768.0
        except Exception as e:
            self.logger.error(f"WAV conversion error: {e}")
            return np.array([])
    
    def _convert_mp3_to_array(self, audio_data: bytes) -> np.ndarray:
        """Convert MP3 bytes to numpy array using basic approach"""
        try:
            # For MP3, we'll try a simple approach
            # Skip MP3 headers and try to extract audio data
            # This is a simplified approach - in production you'd use librosa or pydub
            
            # Try to find audio data after headers
            audio_start = 0
            for i in range(min(1000, len(audio_data) - 2)):
                if audio_data[i:i+2] in [b'\xff\xfb', b'\xff\xf3']:
                    audio_start = i + 100  # Skip header
                    break
            
            if audio_start > 0:
                raw_data = audio_data[audio_start:]
                # Convert to audio assuming 16-bit format
                if len(raw_data) >= 2:
                    audio_array = np.frombuffer(raw_data, dtype=np.int16)
                    return audio_array.astype(np.float32) / 32768.0
            
            return np.array([])
            
        except Exception as e:
            self.logger.error(f"MP3 conversion error: {e}")
            return np.array([])
    
    def _convert_raw_to_array(self, audio_data: bytes) -> np.ndarray:
        """Convert raw bytes to audio array with multiple attempts"""
        try:
            self.logger.info(f"Trying raw conversion with {len(audio_data)} bytes")
            
            # Try different interpretations
            attempts = [
                (np.int16, "int16"),
                (np.int8, "int8"), 
                (np.uint8, "uint8"),
                (np.float32, "float32")
            ]
            
            for dtype, name in attempts:
                try:
                    if len(audio_data) >= np.dtype(dtype).itemsize:
                        # Ensure length is multiple of dtype size
                        trim_length = (len(audio_data) // np.dtype(dtype).itemsize) * np.dtype(dtype).itemsize
                        trimmed_data = audio_data[:trim_length]
                        
                        audio_array = np.frombuffer(trimmed_data, dtype=dtype)
                        
                        if len(audio_array) > 0:
                            # Normalize to float32 range [-1, 1]
                            if dtype == np.int16:
                                normalized = audio_array.astype(np.float32) / 32768.0
                            elif dtype == np.int8:
                                normalized = audio_array.astype(np.float32) / 128.0
                            elif dtype == np.uint8:
                                normalized = (audio_array.astype(np.float32) - 128) / 128.0
                            else:  # float32
                                normalized = audio_array.astype(np.float32)
                            
                            # Basic sanity check - audio should have some variation
                            if np.std(normalized) > 0.001:  # Some actual audio content
                                self.logger.info(f"Successfully converted as {name}: {len(normalized)} samples, std: {np.std(normalized):.6f}")
                                return normalized
                            
                except Exception as e:
                    self.logger.debug(f"Failed {name} conversion: {e}")
                    continue
            
            self.logger.warning("All raw conversion attempts failed")
            return np.array([])
            
        except Exception as e:
            self.logger.error(f"Raw conversion error: {e}")
            return np.array([])
    
    def _analyze_amplitude_simple(self, audio_array: np.ndarray) -> float:
        """Simple amplitude analysis without external libraries"""
        if len(audio_array) == 0:
            return 0.0
        
        # RMS amplitude
        rms = np.sqrt(np.mean(audio_array ** 2))
        
        # Normalize and enhance for better visual response
        normalized = np.clip(rms * 8, 0.0, 1.0)
        
        # Apply power curve for more natural jaw movement
        return normalized ** 0.7
    
    def _analyze_frequency_simple(self, audio_array: np.ndarray) -> float:
        """Simple frequency analysis using basic FFT"""
        if len(audio_array) < 512:
            return 0.0
        
        try:
            # Use smaller window for Pi performance
            window_size = min(512, len(audio_array))
            window = audio_array[:window_size]
            
            # Simple FFT
            fft = np.fft.rfft(window)
            magnitude = np.abs(fft)
            
            if len(magnitude) < 10:
                return 0.0
            
            # Find dominant frequency in speech range
            freqs = np.fft.rfftfreq(window_size, 1/self.sample_rate)
            
            # Focus on speech frequencies (100-400 Hz)
            speech_mask = (freqs >= 100) & (freqs <= 400)
            if not np.any(speech_mask):
                return 0.0
            
            speech_magnitude = magnitude[speech_mask]
            speech_freqs = freqs[speech_mask]
            
            # Weighted average of frequencies
            if np.sum(speech_magnitude) > 0:
                avg_freq = np.average(speech_freqs, weights=speech_magnitude)
                # Normalize to 0-1 range
                normalized = (avg_freq - 100) / 300
                return np.clip(normalized, 0.0, 1.0)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _analyze_speech_energy(self, audio_array: np.ndarray) -> float:
        """Analyze speech energy for micro-movements"""
        if len(audio_array) < 256:
            return 0.0
        
        # Simple energy analysis using windowed approach
        window_size = 128
        energy_values = []
        
        for i in range(0, len(audio_array) - window_size, window_size // 2):
            window = audio_array[i:i + window_size]
            energy = np.sum(window ** 2)
            energy_values.append(energy)
        
        if len(energy_values) < 2:
            return 0.0
        
        # Measure energy variation (speech activity)
        energy_std = np.std(energy_values)
        energy_mean = np.mean(energy_values)
        
        if energy_mean > 0:
            variation = energy_std / energy_mean
            return np.clip(variation * 2, 0.0, 1.0)
        
        return 0.0
    
    def _smooth_feature(self, new_value: float, history: List[float]) -> float:
        """Smooth features for natural movement"""
        history.append(new_value)
        
        if len(history) > self.smoothing_window:
            history.pop(0)
        
        # Simple moving average with slight bias toward recent values
        if len(history) == 1:
            return new_value
        
        weights = np.linspace(0.7, 1.0, len(history))
        smoothed = np.average(history, weights=weights)
        
        return smoothed
    
    def _map_amplitude_to_jaw(self, amplitude: float) -> float:
        """Map audio amplitude to jaw drop (deepfake-like)"""
        min_jaw, max_jaw = self.jaw_range
        # Use power curve for more natural jaw movement
        jaw_amount = amplitude ** 0.8
        return min_jaw + jaw_amount * (max_jaw - min_jaw)
    
    def _map_frequency_to_lip_width(self, frequency: float) -> float:
        """Map frequency to lip width (higher freq = wider lips)"""
        min_width, max_width = self.lip_width_range
        return min_width + frequency * (max_width - min_width)
    
    def _map_amplitude_to_lip_height(self, amplitude: float) -> float:
        """Map amplitude to lip height (louder = more open)"""
        min_height, max_height = self.lip_height_range
        # More aggressive mapping for dramatic effect
        height_amount = amplitude ** 0.6
        return min_height + height_amount * (max_height - min_height)
    
    def _generate_micro_movement(self, speech_energy: float) -> Tuple[float, float]:
        """Generate natural micro-movements"""
        # Scale movement by speech activity
        movement_scale = speech_energy * 1.5
        
        # Add slight random variation
        offset_x = (np.random.random() - 0.5) * movement_scale
        offset_y = (np.random.random() - 0.5) * movement_scale * 0.6
        
        return (offset_x, offset_y)
    
    def _apply_deepfake_deformation(self, face_image: np.ndarray, 
                                  jaw_drop: float, lip_width: float, 
                                  lip_height: float, micro_movement: Tuple[float, float]) -> np.ndarray:
        """Apply realistic mouth deformation like a deepfake video"""
        try:
            h, w = face_image.shape[:2]
            
            # Get mouth center and micro movements
            mouth_x, mouth_y = self.mouth_center
            offset_x, offset_y = micro_movement
            
            # Define mouth region for deformation
            mouth_width = int(w * 0.25)  # Larger region for better deformation
            mouth_height = int(w * 0.20)
            
            # Calculate mouth region bounds
            x1 = max(0, mouth_x - mouth_width // 2)
            x2 = min(w, mouth_x + mouth_width // 2)
            y1 = max(0, mouth_y - mouth_height // 2)
            y2 = min(h, mouth_y + mouth_height // 2)
            
            if x2 <= x1 or y2 <= y1:
                return face_image
            
            # Create control points for mouth deformation
            mouth_region_w = x2 - x1
            mouth_region_h = y2 - y1
            center_x = mouth_region_w // 2
            center_y = mouth_region_h // 2
            
            # Original control points (mouth closed/neutral)
            src_points = np.array([
                # Upper lip points
                [center_x - mouth_region_w//4, center_y - mouth_region_h//6],    # Left upper
                [center_x, center_y - mouth_region_h//8],                       # Center upper
                [center_x + mouth_region_w//4, center_y - mouth_region_h//6],   # Right upper
                
                # Lower lip points  
                [center_x - mouth_region_w//4, center_y + mouth_region_h//6],   # Left lower
                [center_x, center_y + mouth_region_h//8],                       # Center lower
                [center_x + mouth_region_w//4, center_y + mouth_region_h//6],   # Right lower
                
                # Mouth corners
                [center_x - mouth_region_w//3, center_y],                       # Left corner
                [center_x + mouth_region_w//3, center_y],                       # Right corner
            ], dtype=np.float32)
            
            # Deformed control points based on audio analysis
            dst_points = src_points.copy()
            
            # Apply jaw drop (mouth opening)
            jaw_movement = int(jaw_drop * 0.8)  # Scale down for realistic movement
            
            # Move lower lip down for jaw drop
            dst_points[3, 1] += jaw_movement  # Left lower
            dst_points[4, 1] += jaw_movement  # Center lower  
            dst_points[5, 1] += jaw_movement  # Right lower
            
            # Apply lip width changes (wider/narrower mouth)
            width_factor = (lip_width - 1.0) * 0.5  # Scale for subtle effect
            width_adjustment = int(mouth_region_w * width_factor * 0.3)
            
            # Adjust corner positions for width
            dst_points[6, 0] -= width_adjustment  # Left corner
            dst_points[7, 0] += width_adjustment  # Right corner
            
            # Adjust upper and lower lip corners
            dst_points[0, 0] -= width_adjustment // 2  # Left upper
            dst_points[2, 0] += width_adjustment // 2  # Right upper
            dst_points[3, 0] -= width_adjustment // 2  # Left lower
            dst_points[5, 0] += width_adjustment // 2  # Right lower
            
            # Apply lip height changes (pucker effect)
            height_factor = (lip_height - 1.0) * 0.3
            height_adjustment = int(mouth_region_h * height_factor * 0.4)
            
            # Move upper lip points
            dst_points[0, 1] -= height_adjustment // 2  # Left upper
            dst_points[1, 1] -= height_adjustment      # Center upper
            dst_points[2, 1] -= height_adjustment // 2  # Right upper
            
            # Apply micro movements for natural variation
            for i in range(len(dst_points)):
                dst_points[i, 0] += offset_x * 0.5
                dst_points[i, 1] += offset_y * 0.5
            
            # Extract mouth region
            mouth_region = face_image[y1:y2, x1:x2].copy()
            
            # Create deformation using piecewise affine transformation
            deformed_mouth = self._apply_piecewise_affine_transform(
                mouth_region, src_points, dst_points
            )
            
            # Create sophisticated blending mask
            mask = self._create_video_blend_mask(mouth_region.shape[:2])
            
            # Blend deformed mouth back into face
            result_face = face_image.copy()
            
            # Multi-channel blending for seamless integration
            for c in range(3):
                result_face[y1:y2, x1:x2, c] = (
                    mask * deformed_mouth[:, :, c] + 
                    (1 - mask) * result_face[y1:y2, x1:x2, c]
                )
            
            return result_face.astype(np.uint8)
            
        except Exception as e:
            self.logger.error(f"Error in video-like mouth deformation: {e}")
            return face_image
    
    def _apply_piecewise_affine_transform(self, image: np.ndarray, src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
        """Apply piecewise affine transformation for realistic mouth deformation"""
        try:
            h, w = image.shape[:2]
            
            # Create triangulation for piecewise transformation
            # Add corner points to ensure full coverage
            all_src_points = np.vstack([
                src_points,
                [[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]]  # Image corners
            ])
            
            all_dst_points = np.vstack([
                dst_points,
                [[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]]  # Keep corners fixed
            ])
            
            # Simple grid-based warping for Pi compatibility
            result = image.copy()
            
            # Create a grid of points for smooth deformation
            grid_size = 20
            for y in range(0, h, grid_size):
                for x in range(0, w, grid_size):
                    # Find closest control point and apply its transformation
                    distances = np.sum((all_src_points - [x, y])**2, axis=1)
                    closest_idx = np.argmin(distances)
                    
                    if distances[closest_idx] < (grid_size * 2)**2:  # Within influence
                        # Calculate transformation vector
                        transform_vector = all_dst_points[closest_idx] - all_src_points[closest_idx]
                        
                        # Apply weighted transformation to nearby pixels
                        y_end = min(y + grid_size, h)
                        x_end = min(x + grid_size, w)
                        
                        for py in range(y, y_end):
                            for px in range(x, x_end):
                                # Calculate weight based on distance
                                dist_to_point = np.sqrt((px - x)**2 + (py - y)**2)
                                weight = max(0, 1 - dist_to_point / grid_size)
                                
                                # Apply weighted transformation
                                new_x = int(px + transform_vector[0] * weight)
                                new_y = int(py + transform_vector[1] * weight)
                                
                                # Bounds checking
                                if 0 <= new_x < w and 0 <= new_y < h:
                                    result[py, px] = image[new_y, new_x]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in piecewise affine transform: {e}")
            return image
    
    def _create_video_blend_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Create video-like blending mask for seamless mouth integration"""
        h, w = shape
        
        # Create soft elliptical mask that covers the deformed area
        y, x = np.ogrid[:h, :w]
        center_x, center_y = w // 2, h // 2
        
        # Large soft mask for seamless blending
        mask = ((x - center_x) / (w * 0.4))**2 + ((y - center_y) / (h * 0.4))**2
        mask = 1.0 - np.clip(mask, 0, 1)
        
        # Apply strong blur for video-like seamless blending
        blurred_mask = cv2.GaussianBlur(mask.astype(np.float32), (31, 31), 0)
        
        return np.clip(blurred_mask, 0, 1)
    
    def _add_subtle_face_effects(self, face_image: np.ndarray, jaw_drop: float, lip_height: float) -> np.ndarray:
        """Add subtle overall face effects for enhanced realism"""
        try:
            # Slight brightness variation based on mouth opening (simulates lighting change)
            brightness_factor = 1.0 + (jaw_drop / 100) * 0.02
            
            # Apply subtle gamma correction for more natural look
            gamma = 1.0 + (lip_height - 1.0) * 0.05
            gamma = np.clip(gamma, 0.95, 1.05)
            
            # Apply effects
            enhanced = cv2.convertScaleAbs(face_image, alpha=brightness_factor, beta=0)
            
            # Gamma correction
            gamma_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            enhanced = cv2.LUT(enhanced, gamma_table)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Error in face effects: {e}")
            return face_image
    
    def cleanup(self):
        """Cleanup resources"""
        self.amplitude_history.clear()
        self.frequency_history.clear()
        self.logger.info("Audio-Driven Deepfake Face system cleaned up") 