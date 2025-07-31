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
                self.logger.error(f"Could not load base face from {face_image_path}")
                return False
            
            self.face_height, self.face_width = self.base_face.shape[:2]
            
            # Calculate approximate mouth center (simplified approach for Pi)
            # Assume mouth is in the lower third, center horizontally
            self.mouth_center = (
                self.face_width // 2,  # Center horizontally
                int(self.face_height * 0.75)  # Lower third vertically
            )
            
            # 🔍 DEBUG: Show mouth center calculation
            print(f"🎭 MOUTH CENTER DEBUG: Face loaded - size: {self.face_width}x{self.face_height}")
            print(f"🎭 MOUTH CENTER DEBUG: Calculated mouth center: {self.mouth_center}")
            print(f"🎭 MOUTH CENTER DEBUG: That's {self.mouth_center[0]/self.face_width*100:.1f}% across, {self.mouth_center[1]/self.face_height*100:.1f}% down")
            
            self.logger.info(f"Base face loaded: {self.face_width}x{self.face_height}, mouth center: {self.mouth_center}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading base face: {e}")
            return False
    
    def generate_face_from_audio(self, audio_data: bytes, duration: float) -> np.ndarray:
        """Generate deepfake-like face animation from audio data"""
        try:
            print(f"🎭 GENERATE DEBUG: generate_face_from_audio called with {len(audio_data)} bytes, duration {duration}")
            self.logger.info(f"generate_face_from_audio called with {len(audio_data)} bytes, duration {duration}")
            
            if self.base_face is None:
                print(f"❌ GENERATE DEBUG: Base face is None!")
                self.logger.error("Base face is None!")
                return self.base_face
            
            print(f"✅ GENERATE DEBUG: Base face exists, shape: {self.base_face.shape}")
            
            # Convert audio bytes to numpy array
            audio_array = self._bytes_to_audio_array(audio_data)
            print(f"✅ GENERATE DEBUG: Audio converted to array: {len(audio_array)} samples")
            self.logger.info(f"Converted to audio array: {len(audio_array)} samples")
            
            if len(audio_array) == 0:
                print(f"❌ GENERATE DEBUG: Audio array is empty")
                self.logger.warning("Audio array is empty")
                return self.base_face
            
            # Analyze audio features (simplified for Pi)
            print(f"🔊 GENERATE DEBUG: About to analyze audio features...")
            amplitude = self._analyze_amplitude_simple(audio_array)
            dominant_freq = self._analyze_frequency_simple(audio_array)
            speech_energy = self._analyze_speech_energy(audio_array)
            
            print(f"✅ GENERATE DEBUG: Audio analysis complete - amp:{amplitude:.3f}, freq:{dominant_freq:.3f}, energy:{speech_energy:.3f}")
            self.logger.info(f"Audio analysis - amplitude: {amplitude:.3f}, freq: {dominant_freq:.3f}, energy: {speech_energy:.3f}")
            
            # Smooth features over time for natural movement
            print(f"🔄 GENERATE DEBUG: About to smooth features...")
            smoothed_amplitude = self._smooth_feature(amplitude, self.amplitude_history)
            smoothed_frequency = self._smooth_feature(dominant_freq, self.frequency_history)
            
            print(f"✅ GENERATE DEBUG: Smoothing complete - smooth_amp:{smoothed_amplitude:.3f}, smooth_freq:{smoothed_frequency:.3f}")
            self.logger.info(f"Smoothed - amplitude: {smoothed_amplitude:.3f}, freq: {smoothed_frequency:.3f}")
            
            # Map audio features to facial parameters
            print(f"🎭 GENERATE DEBUG: About to map audio to facial parameters...")
            jaw_drop = self._map_amplitude_to_jaw(smoothed_amplitude)
            lip_width = self._map_frequency_to_lip_width(smoothed_frequency)
            lip_height = self._map_amplitude_to_lip_height(smoothed_amplitude)
            micro_movement = self._generate_micro_movement(speech_energy)
            
            print(f"✅ GENERATE DEBUG: Facial parameters mapped - jaw:{jaw_drop:.1f}, lip_w:{lip_width:.3f}, lip_h:{lip_height:.3f}")
            
            # 🔍 DEBUG: Track parameter changes to verify dynamic movement
            if not hasattr(self, '_param_history'):
                self._param_history = []
            
            current_params = (jaw_drop, lip_width, lip_height, smoothed_amplitude)
            self._param_history.append(current_params)
            
            # Show parameter changes every few frames
            if len(self._param_history) % 3 == 0:  # Every 3rd frame
                print(f"🎭 PARAMS DEBUG: jaw={jaw_drop:.1f}, lip_w={lip_width:.3f}, lip_h={lip_height:.3f}, amp={smoothed_amplitude:.3f}")
                
                # Check if parameters are actually changing
                if len(self._param_history) >= 3:
                    recent_params = self._param_history[-3:]
                    jaw_variance = max([p[0] for p in recent_params]) - min([p[0] for p in recent_params])
                    width_variance = max([p[1] for p in recent_params]) - min([p[1] for p in recent_params])
                    height_variance = max([p[2] for p in recent_params]) - min([p[2] for p in recent_params])
                    
                    print(f"🎭 VARIANCE DEBUG: jaw_var={jaw_variance:.2f}, width_var={width_variance:.3f}, height_var={height_variance:.3f}")
                    
                    if jaw_variance < 0.5 and width_variance < 0.01 and height_variance < 0.01:
                        print("⚠️  STATIC PARAMS WARNING: Parameters are not changing much - mouth may appear static!")
            
            self.logger.info(f"Face params - jaw: {jaw_drop:.1f}, width: {lip_width:.3f}, height: {lip_height:.3f}, movement: {micro_movement}")
            
            # Generate deepfake-like face with these parameters
            print(f"🎭 GENERATE DEBUG: About to apply deepfake deformation...")
            deepfake_face = self._apply_deepfake_deformation(
                self.base_face.copy(),
                jaw_drop, lip_width, lip_height, micro_movement
            )
            
            print(f"✅ GENERATE DEBUG: Deformation complete, returning face with shape: {deepfake_face.shape}")
            self.logger.info(f"Generated deepfake face with shape: {deepfake_face.shape}")
            return deepfake_face
            
        except Exception as e:
            print(f"❌ GENERATE DEBUG: Major error in generate_face_from_audio: {e}")
            import traceback
            print(f"❌ GENERATE TRACEBACK: {traceback.format_exc()}")
            self.logger.error(f"Error generating deepfake face: {e}")
            return self.base_face if self.base_face is not None else np.zeros((512, 512, 3), dtype=np.uint8)
    
    def _bytes_to_audio_array(self, audio_data: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array (Pi-compatible)"""
        try:
            print(f"🎵 AUDIO CONVERSION: Converting {len(audio_data)} bytes")
            print(f"🎵 AUDIO CONVERSION: First 10 bytes: {audio_data[:10]}")
            self.logger.info(f"Audio data starts with: {audio_data[:10]}")
            
            # Try different audio format detection
            if audio_data.startswith(b'RIFF'):
                # WAV format
                print("🎵 AUDIO CONVERSION: Detected WAV format")
                self.logger.info("Detected WAV format")
                result = self._convert_wav_to_array(audio_data)
                print(f"✅ AUDIO CONVERSION: WAV result: {len(result)} samples")
                return result
            elif audio_data.startswith(b'\xff\xfb') or audio_data.startswith(b'\xff\xf3'):
                # MP3 format
                print("🎵 AUDIO CONVERSION: Detected MP3 format")
                self.logger.info("Detected MP3 format - converting")
                result = self._convert_mp3_to_array(audio_data)
                print(f"✅ AUDIO CONVERSION: MP3 result: {len(result)} samples")
                return result
            else:
                # Try to detect other formats or raw audio
                print("🎵 AUDIO CONVERSION: Unknown format, trying raw")
                self.logger.info("Unknown format, trying raw audio conversion")
                result = self._convert_raw_to_array(audio_data)
                print(f"✅ AUDIO CONVERSION: Raw result: {len(result)} samples")
                return result
                
        except Exception as e:
            print(f"❌ AUDIO CONVERSION: Failed: {e}")
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
    
    def _analyze_amplitude_simple(self, audio_chunk: np.ndarray) -> float:
        """Simple amplitude analysis for Pi compatibility"""
        if len(audio_chunk) == 0:
            return 0.0
        
        # Calculate RMS (Root Mean Square) amplitude
        rms = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))
        
        # Debug output (occasionally)
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        if self._debug_counter % 10 == 0:  # Every 10th call
            print(f"🔊 AMPLITUDE: chunk_size={len(audio_chunk)}, rms={rms:.4f}")
        
        # 🔧 FIX: Better normalization for more dynamic movement
        # Instead of normalizing to max, use a more reasonable scale
        # Most audio will be between 0.0 and 0.8, so let's scale accordingly
        normalized = np.clip(rms / 0.6, 0.0, 1.0)  # Scale by 0.6 instead of max
        
        # Apply a curve to make movements more dramatic
        # Use a power curve to enhance smaller movements
        result = normalized ** 0.7  # Power curve for better dynamics
        
        if self._debug_counter % 10 == 0:  # Every 10th call
            print(f"🔊 AMPLITUDE: normalized={normalized:.4f}, result={result:.4f}")
        
        return result
    
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
        """Map audio amplitude to jaw drop amount"""
        min_jaw, max_jaw = self.jaw_range
        jaw_drop = min_jaw + (max_jaw - min_jaw) * amplitude
        
        # 🔧 MAKE MORE DRAMATIC: Increase jaw movement for visibility  
        jaw_drop = jaw_drop * 2.0  # Double the jaw movement
        
        return jaw_drop
    
    def _map_frequency_to_lip_width(self, frequency: float) -> float:
        """Map dominant frequency to lip width"""
        min_width, max_width = self.lip_width_range
        lip_width = min_width + (max_width - min_width) * frequency
        
        # 🔧 MAKE MORE DRAMATIC: Increase lip width variation
        # Make it vary more dramatically between 0.7 and 1.4
        lip_width = 0.7 + (1.4 - 0.7) * frequency
        
        return lip_width
    
    def _map_amplitude_to_lip_height(self, amplitude: float) -> float:
        """Map audio amplitude to lip height"""
        min_height, max_height = self.lip_height_range
        lip_height = min_height + (max_height - min_height) * amplitude
        
        # 🔧 MAKE MORE DRAMATIC: Increase lip height variation  
        lip_height = lip_height * 1.5  # Make lip height changes more dramatic
        
        return lip_height
    
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
            # 🔍 DEBUG: Track what's happening with deformation
            print(f"🎭 DEFORM DEBUG: Starting deformation - jaw:{jaw_drop:.2f}, lip_w:{lip_width:.3f}, lip_h:{lip_height:.3f}")
            
            h, w = face_image.shape[:2]
            
            # Get mouth center and micro movements
            mouth_x, mouth_y = self.mouth_center
            offset_x, offset_y = micro_movement
            
            print(f"🎭 DEFORM DEBUG: Face shape: {face_image.shape}, mouth center: ({mouth_x}, {mouth_y})")
            
            # Define mouth region for deformation (smaller for better performance)
            mouth_width = int(w * 0.15)  
            mouth_height = int(w * 0.12)
            
            # Calculate mouth region bounds
            x1 = max(0, mouth_x - mouth_width // 2)
            x2 = min(w, mouth_x + mouth_width // 2)
            y1 = max(0, mouth_y - mouth_height // 2)
            y2 = min(h, mouth_y + mouth_height // 2)
            
            print(f"🎭 DEFORM DEBUG: Mouth region: ({x1},{y1}) to ({x2},{y2}), size: {mouth_width}x{mouth_height}")
            
            if x2 <= x1 or y2 <= y1:
                print("❌ DEFORM DEBUG: Invalid mouth region, returning original")
                return face_image
            
            # Extract mouth region
            mouth_region = face_image[y1:y2, x1:x2].copy()
            mouth_h, mouth_w = mouth_region.shape[:2]
            
            if mouth_h < 10 or mouth_w < 10:  # Too small to deform
                print(f"❌ DEFORM DEBUG: Mouth region too small: {mouth_w}x{mouth_h}")
                return face_image
            
            # Create simpler transformation matrix for reliable Pi performance
            center_x, center_y = mouth_w // 2, mouth_h // 2
            
            # Scale factors based on audio analysis
            scale_x = lip_width
            scale_y = lip_height
            
            # Translation for jaw drop and micro movements
            translate_x = offset_x * 0.5
            translate_y = jaw_drop * 0.3 + offset_y * 0.5
            
            print(f"🎭 DEFORM DEBUG: Transform - scale_x:{scale_x:.3f}, scale_y:{scale_y:.3f}, translate_x:{translate_x:.2f}, translate_y:{translate_y:.2f}")
            
            # Simple affine transformation matrix
            transform_matrix = np.array([
                [scale_x, 0, center_x * (1 - scale_x) + translate_x],
                [0, scale_y, center_y * (1 - scale_y) + translate_y]
            ], dtype=np.float32)
            
            # Apply transformation with OpenCV (much more reliable than custom code)
            deformed_mouth = cv2.warpAffine(
                mouth_region, 
                transform_matrix, 
                (mouth_w, mouth_h),
                flags=cv2.INTER_LINEAR,  # Faster than INTER_CUBIC
                borderMode=cv2.BORDER_REFLECT
            )
            
            print(f"✅ DEFORM DEBUG: Mouth deformation completed successfully")
            
            # Create blending mask for seamless integration
            mask = self._create_simple_blend_mask(mouth_region.shape[:2])
            
            # Blend deformed mouth back into face
            result_face = face_image.copy()
            
            # Ensure dimensions match before blending
            target_region = result_face[y1:y2, x1:x2]
            
            # Verify all dimensions match
            if (deformed_mouth.shape != target_region.shape or 
                mask.shape != deformed_mouth.shape[:2]):
                self.logger.warning(f"Dimension mismatch: deformed={deformed_mouth.shape}, target={target_region.shape}, mask={mask.shape}")
                # Resize mask if needed
                if mask.shape != deformed_mouth.shape[:2]:
                    mask = cv2.resize(mask, (deformed_mouth.shape[1], deformed_mouth.shape[0]))
                # If still mismatched, skip blending
                if deformed_mouth.shape != target_region.shape:
                    print("❌ DEFORM DEBUG: Dimension mismatch after resize, returning original")
                    return face_image
            
            # Apply blending with proper dimension checks
            try:
                for c in range(min(3, deformed_mouth.shape[2], target_region.shape[2])):
                    result_face[y1:y2, x1:x2, c] = (
                        mask * deformed_mouth[:, :, c] + 
                        (1 - mask) * target_region[:, :, c]
                    )
                print(f"✅ DEFORM DEBUG: Blending completed - result shape: {result_face.shape}")
            except Exception as blend_error:
                print(f"❌ DEFORM DEBUG: Blending error: {blend_error}")
                self.logger.error(f"Blending error: {blend_error}")
                return face_image
            
            # Ensure result_face is valid before conversion
            try:
                # Check for any invalid values
                if not np.isfinite(result_face).all():
                    self.logger.warning("Result face contains invalid values, cleaning...")
                    result_face = np.nan_to_num(result_face, nan=0, posinf=255, neginf=0)
                
                # Ensure values are in valid range for uint8
                result_face = np.clip(result_face, 0, 255)
                
                # Convert to uint8 safely
                print(f"✅ DEFORM DEBUG: Final result ready - returning deformed face")
                return result_face.astype(np.uint8)
                
            except Exception as conversion_error:
                print(f"❌ DEFORM DEBUG: Conversion error: {conversion_error}")
                self.logger.error(f"Error converting result face: {conversion_error}")
                self.logger.error(f"Result face info: shape={result_face.shape}, dtype={result_face.dtype}, min={np.min(result_face)}, max={np.max(result_face)}")
                # Return original face as fallback
                return face_image.astype(np.uint8)
            
        except Exception as e:
            print(f"❌ DEFORM DEBUG: Major error in deformation: {e}")
            self.logger.error(f"Error in simplified mouth deformation: {e}")
            return face_image
    
    def _create_simple_blend_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Create simple, fast blending mask"""
        h, w = shape
        
        # Create simple elliptical mask
        y, x = np.ogrid[:h, :w]
        center_x, center_y = w // 2, h // 2
        
        # Simple ellipse equation
        mask = ((x - center_x) / (w * 0.45))**2 + ((y - center_y) / (h * 0.45))**2
        mask = 1.0 - np.clip(mask, 0, 1)
        
        # Light blur for smooth edges (smaller kernel for speed)
        blurred_mask = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 0)
        
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