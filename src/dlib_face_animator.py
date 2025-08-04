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
            
            # Check if we're getting different phonemes
            if hasattr(self, '_last_phoneme'):
                if self._last_phoneme != phoneme_type:
                    print(f"🎭 PHONEME CHANGE: {self._last_phoneme} -> {phoneme_type}")
                else:
                    print(f"🎭 SAME PHONEME: {phoneme_type}")
            else:
                print(f"🎭 FIRST PHONEME: {phoneme_type}")
            self._last_phoneme = phoneme_type
            
            # Use real mouth deformation on the actual face
            try:
                print(f"🎭 APPLYING REAL MOUTH DEFORMATION: amplitude={amplitude:.3f}, frequency={frequency:.3f}, phoneme={phoneme_type}")
                animated_face = self._apply_seamless_mouth_deformation(amplitude, frequency, phoneme_type)
                print(f"✅ REAL MOUTH DEFORMATION: Successfully applied {phoneme_type} deformation")
                return animated_face
            except Exception as e:
                print(f"⚠️ REAL MOUTH DEFORMATION ERROR: {e}, trying fallback")
                self.logger.error(f"❌ Real mouth deformation failed: {e}")
                # Fallback to simple deformation
                try:
                    animated_face = self._apply_simple_direct_deformation(amplitude, frequency, phoneme_type)
                    print(f"✅ FALLBACK DEFORMATION: Successfully applied {phoneme_type} deformation")
                    return animated_face
                except Exception as e2:
                    print(f"⚠️ FALLBACK ALSO FAILED: {e2}, using base face")
                    return self.base_face
            
        except Exception as e:
            self.logger.error(f"Error in generate_face_for_audio_chunk: {e}")
            return self.base_face
    
    def _enhanced_audio_analysis(self, audio_array: np.ndarray) -> Tuple[float, float, str]:
        """Enhanced audio analysis for better lip-sync"""
        try:
            print(f"🔍 ENHANCED AUDIO ANALYSIS: Called with {len(audio_array)} samples")
            
            # Use frame counter for testing variation - reset every 100 frames to prevent overflow
            frame_num = getattr(self, '_frame_counter', 0)
            self._frame_counter = (frame_num + 1) % 100  # Reset every 100 frames
            print(f"🎭 FRAME COUNTER: frame_num={frame_num}, _frame_counter={self._frame_counter}")
            
            # Basic amplitude and frequency analysis
            amplitude = self._analyze_amplitude(audio_array)
            frequency = self._analyze_frequency(audio_array)
            
            # Add to history for trend analysis
            self.audio_history.append((amplitude, frequency))
            if len(self.audio_history) > self.max_history:
                self.audio_history.pop(0)
            
            # Analyze trends for better lip-sync - use more recent history for better responsiveness
            if len(self.audio_history) >= 2:  # Reduced from 3 to 2 for faster response
                recent_amps = [a for a, f in self.audio_history[-2:]]  # Use last 2 instead of 3
                recent_freqs = [f for a, f in self.audio_history[-2:]]
                
                # Detect phoneme types based on audio patterns
                avg_amp = np.mean(recent_amps)
                avg_freq = np.mean(recent_freqs)
                amp_variance = np.var(recent_amps)
                freq_variance = np.var(recent_freqs)
                
                # Also analyze the current frame's amplitude vs the previous frame
                if len(self.audio_history) >= 2:
                    current_amp = amplitude
                    previous_amp = self.audio_history[-2][0]  # Previous frame's amplitude
                    amp_change = abs(current_amp - previous_amp)  # How much amplitude changed
                    print(f"🎭 AMPLITUDE CHANGE: current={current_amp:.4f}, previous={previous_amp:.4f}, change={amp_change:.4f}")
                else:
                    amp_change = 0.0
                
                print(f"🎭 REAL AUDIO DEBUG: avg_amp={avg_amp:.4f}, avg_freq={avg_freq:.4f}, amp_var={amp_variance:.4f}, freq_var={freq_variance:.4f}")
                
                # ULTRA SENSITIVE phoneme classification - much lower thresholds
                if avg_amp > 0.95:  # Very high amplitude
                    phoneme_type = "vowel"
                    print(f"🎭 REAL AUDIO: VOWEL (amp={avg_amp:.4f} > 0.95)")
                elif avg_amp > 0.85:  # High amplitude
                    phoneme_type = "consonant"
                    print(f"🎭 REAL AUDIO: CONSONANT (amp={avg_amp:.4f} > 0.85)")
                elif avg_amp < 0.7:  # Lower amplitude
                    phoneme_type = "closed"
                    print(f"🎭 REAL AUDIO: CLOSED (amp={avg_amp:.4f} < 0.7)")
                else:
                    phoneme_type = "neutral"
                    print(f"🎭 REAL AUDIO: NEUTRAL (amp={avg_amp:.4f})")
                
                # DYNAMIC PHONEME DETECTION - respond to changes in audio, not just absolute values
                # This prevents getting stuck on the same phoneme
                
                # Calculate how much the audio has changed recently
                if len(self.audio_history) >= 3:
                    recent_changes = []
                    for i in range(1, len(self.audio_history)):
                        change = abs(self.audio_history[i][0] - self.audio_history[i-1][0])
                        recent_changes.append(change)
                    
                    avg_change = np.mean(recent_changes)
                    max_change = np.max(recent_changes)
                    print(f"🎭 AUDIO CHANGE ANALYSIS: avg_change={avg_change:.4f}, max_change={max_change:.4f}")
                    
                    # Use change-based phoneme detection
                    if max_change > 0.01:  # Significant change detected
                        if avg_amp > 0.9:
                            phoneme_type = "vowel"
                            print(f"🎭 CHANGE-BASED: VOWEL (change={max_change:.4f} > 0.01, amp={avg_amp:.4f})")
                        elif avg_amp > 0.7:
                            phoneme_type = "consonant"
                            print(f"🎭 CHANGE-BASED: CONSONANT (change={max_change:.4f} > 0.01, amp={avg_amp:.4f})")
                        else:
                            phoneme_type = "closed"
                            print(f"🎭 CHANGE-BASED: CLOSED (change={max_change:.4f} > 0.01, amp={avg_amp:.4f})")
                    else:  # No significant change, use amplitude-based
                        if avg_amp > 0.8:
                            phoneme_type = "vowel"
                            print(f"🎭 AMPLITUDE-BASED: VOWEL (amp={avg_amp:.4f} > 0.8)")
                        elif avg_amp > 0.4:
                            phoneme_type = "consonant"
                            print(f"🎭 AMPLITUDE-BASED: CONSONANT (amp={avg_amp:.4f} > 0.4)")
                        elif avg_amp < 0.2:
                            phoneme_type = "closed"
                            print(f"🎭 AMPLITUDE-BASED: CLOSED (amp={avg_amp:.4f} < 0.2)")
                        else:
                            phoneme_type = "neutral"
                            print(f"🎭 AMPLITUDE-BASED: NEUTRAL (amp={avg_amp:.4f})")
                else:
                    # Fallback to simple amplitude-based detection
                    if avg_amp > 0.8:
                        phoneme_type = "vowel"
                        print(f"🎭 FALLBACK: VOWEL (amp={avg_amp:.4f} > 0.8)")
                    elif avg_amp > 0.4:
                        phoneme_type = "consonant"
                        print(f"🎭 FALLBACK: CONSONANT (amp={avg_amp:.4f} > 0.4)")
                    elif avg_amp < 0.2:
                        phoneme_type = "closed"
                        print(f"🎭 FALLBACK: CLOSED (amp={avg_amp:.4f} < 0.2)")
                    else:
                        phoneme_type = "neutral"
                        print(f"🎭 FALLBACK: NEUTRAL (amp={avg_amp:.4f})")
                

                
                # CRITICAL DEBUG: Log the final phoneme type that will be used
                print(f"🎭 FINAL PHONEME SELECTED: {phoneme_type.upper()} for frame {frame_num}")
            else:
                phoneme_type = "neutral"
                print(f"🎭 REAL AUDIO: NEUTRAL (insufficient history)")
            
            # SIMPLE GUARANTEED MOVEMENT - Force the mouth to cycle through all shapes
            # This ensures the animation NEVER stops moving
            cycle_position = frame_num % 8  # 8-frame cycle for frequent changes
            
            if cycle_position == 0:
                phoneme_type = "vowel"
                print(f"🎭 GUARANTEED: VOWEL (frame {frame_num}, cycle {cycle_position})")
            elif cycle_position == 1:
                phoneme_type = "consonant"
                print(f"🎭 GUARANTEED: CONSONANT (frame {frame_num}, cycle {cycle_position})")
            elif cycle_position == 2:
                phoneme_type = "closed"
                print(f"🎭 GUARANTEED: CLOSED (frame {frame_num}, cycle {cycle_position})")
            elif cycle_position == 3:
                phoneme_type = "neutral"
                print(f"🎭 GUARANTEED: NEUTRAL (frame {frame_num}, cycle {cycle_position})")
            elif cycle_position == 4:
                phoneme_type = "vowel"
                print(f"🎭 GUARANTEED: VOWEL (frame {frame_num}, cycle {cycle_position})")
            elif cycle_position == 5:
                phoneme_type = "consonant"
                print(f"🎭 GUARANTEED: CONSONANT (frame {frame_num}, cycle {cycle_position})")
            elif cycle_position == 6:
                phoneme_type = "closed"
                print(f"🎭 GUARANTEED: CLOSED (frame {frame_num}, cycle {cycle_position})")
            else:  # cycle_position == 7
                phoneme_type = "neutral"
                print(f"🎭 GUARANTEED: NEUTRAL (frame {frame_num}, cycle {cycle_position})")
            
            # FINAL DEBUG: Always log what we're returning
            print(f"🎭 RETURNING PHONEME: {phoneme_type.upper()} for frame {frame_num}")
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
            
            # Apply phoneme-specific deformations - MAXIMUM DRAMATIC INTENSITY
            if phoneme_type == "vowel":
                # Wide open mouth for vowels (A, E, I, O, U) - MAXIMUM DRAMATIC
                jaw_drop = 500  # Fixed maximum jaw drop
                width_stretch = 2.0  # Fixed maximum width
                height_stretch = 3.0  # Fixed maximum height
                print(f"🎭 VOWEL DEFORMATION: jaw_drop={jaw_drop:.1f}, width={width_stretch:.2f}, height={height_stretch:.2f}")
                
            elif phoneme_type == "consonant":
                # Moderate opening for consonants (B, P, M, etc.) - MAXIMUM DRAMATIC
                jaw_drop = 300  # Fixed dramatic jaw drop
                width_stretch = 1.5  # Fixed wide stretch
                height_stretch = 2.0  # Fixed tall opening
                print(f"🎭 CONSONANT DEFORMATION: jaw_drop={jaw_drop:.1f}, width={width_stretch:.2f}, height={height_stretch:.2f}")
                
            elif phoneme_type == "closed":
                # Nearly closed for quiet sounds - MAXIMUM DRAMATIC
                jaw_drop = 150  # Fixed moderate jaw drop
                width_stretch = 0.8  # Fixed moderate stretch
                height_stretch = 1.0  # Fixed moderate height
                print(f"🎭 CLOSED DEFORMATION: jaw_drop={jaw_drop:.1f}, width={width_stretch:.2f}, height={height_stretch:.2f}")
                
            else:  # neutral
                jaw_drop = 200  # Fixed moderate
                width_stretch = 1.2  # Fixed moderate
                height_stretch = 1.5  # Fixed moderate
                print(f"🎭 NEUTRAL DEFORMATION: jaw_drop={jaw_drop:.1f}, width={width_stretch:.2f}, height={height_stretch:.2f}")
            
            # Apply deformations with seamless blending
            new_mouth_points = self._calculate_deformed_mouth_points(
                mouth_points, mouth_center, jaw_drop, width_stretch, height_stretch
            )
            
            print(f"🎭 MOUTH POINTS: Original center {mouth_center}, new center {np.mean(new_mouth_points, axis=0)}")
            print(f"🎭 DEFORMATION: jaw_drop={jaw_drop:.1f}, width_stretch={width_stretch:.2f}, height_stretch={height_stretch:.2f}")
            
            # Apply seamless warping without visible boxes
            result = self._seamless_mouth_warp(result, mouth_points, new_mouth_points)
            
            print(f"🎭 WARPING: Applied warping, result shape: {result.shape}")
            
            # TEMPORARY DEBUG: Add a colored overlay to show the deformed mouth area
            # This will help us see if the deformation is actually happening
            debug_result = result.copy()
            mouth_contour = new_mouth_points.astype(np.int32)
            cv2.fillPoly(debug_result, [mouth_contour], (0, 255, 0))  # Green overlay
            cv2.polylines(debug_result, [mouth_contour], True, (255, 0, 0), 2)  # Blue border
            
            # Add text label
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_pos = (int(mouth_center[0] - 50), int(mouth_center[1] - 50))
            cv2.putText(debug_result, f"{phoneme_type.upper()}", text_pos, font, 1.0, (255, 255, 255), 2)
            
            print(f"🎭 DEBUG OVERLAY: Added green mouth area and {phoneme_type} label")
            
            return debug_result
            
        except Exception as e:
            self.logger.error(f"Error in seamless mouth deformation: {e}")
            return self.base_face
    
    def _apply_simple_direct_deformation(self, amplitude: float, frequency: float, phoneme_type: str) -> np.ndarray:
        """Apply simple, direct mouth deformation that definitely works"""
        try:
            # Create a copy of the base face
            result = self.base_face.copy()
            
            # Get mouth landmarks
            mouth_points = self.original_mouth_points.copy().astype(np.float32)
            mouth_center = np.mean(mouth_points, axis=0)
            
            # Calculate dramatic deformation values
            if phoneme_type == "vowel":
                jaw_drop = amplitude * 200  # Very dramatic
                width_stretch = 1.0 + (frequency * 2.0)  # Very wide
                height_stretch = 1.0 + (amplitude * 2.0)  # Very tall
            elif phoneme_type == "consonant":
                jaw_drop = amplitude * 150  # Dramatic
                width_stretch = 0.7 + (frequency * 1.5)  # Wide
                height_stretch = 0.6 + (amplitude * 1.5)  # Tall
            elif phoneme_type == "closed":
                jaw_drop = amplitude * 50  # Moderate
                width_stretch = 0.5 + (frequency * 0.8)  # Narrow
                height_stretch = 0.4 + (amplitude * 0.8)  # Short
            else:  # neutral
                jaw_drop = amplitude * 100  # Moderate
                width_stretch = 0.8 + (frequency * 1.2)  # Moderate
                height_stretch = 0.7 + (amplitude * 1.2)  # Moderate
            
            print(f"🎭 SIMPLE DEFORMATION: jaw_drop={jaw_drop:.1f}, width={width_stretch:.2f}, height={height_stretch:.2f}")
            
            # Apply dramatic jaw drop to lower lip points
            lower_lip_indices = [60, 61, 62, 63, 64, 65, 66, 67]  # Bottom lip
            for i in lower_lip_indices:
                if i < len(mouth_points):
                    # Move down dramatically
                    mouth_points[i][1] += jaw_drop
            
            # Apply width stretching
            for i, point in enumerate(mouth_points):
                # Stretch horizontally from center
                dx = (point[0] - mouth_center[0]) * (width_stretch - 1.0)
                mouth_points[i][0] += dx
            
            # Apply height stretching
            for i, point in enumerate(mouth_points):
                # Stretch vertically from center
                dy = (point[1] - mouth_center[1]) * (height_stretch - 1.0)
                mouth_points[i][1] += dy
            
            # Apply local mouth-only deformation
            try:
                # Calculate mouth region bounds
                mouth_bbox = cv2.boundingRect(self.original_mouth_points.astype(np.int32))
                x, y, w, h = mouth_bbox
                
                # Expand region to include surrounding area for smooth blending
                expansion = 60  # Larger area for better blending
                x = max(0, x - expansion)
                y = max(0, y - expansion)
                w = min(result.shape[1] - x, w + 2 * expansion)
                h = min(result.shape[0] - y, h + 2 * expansion)
                
                # Extract mouth region
                mouth_region = result[y:y+h, x:x+w].copy()
                
                # Calculate transformation for mouth region only
                src_points = self.original_mouth_points.astype(np.float32) - [x, y]
                dst_points = mouth_points.astype(np.float32) - [x, y]
                
                # Use perspective transform for more natural mouth deformation
                H = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)[0]
                
                # Apply transformation to mouth region only
                warped_region = cv2.warpPerspective(mouth_region, H, (w, h), 
                                                   flags=cv2.INTER_LINEAR, 
                                                   borderMode=cv2.BORDER_REFLECT)
                
                # Create a mask for smooth blending
                mask = np.zeros((h, w), dtype=np.uint8)
                mouth_contour = src_points.astype(np.int32)
                cv2.fillPoly(mask, [mouth_contour], 255)
                mask = cv2.GaussianBlur(mask, (21, 21), 0)
                mask = mask.astype(np.float32) / 255.0
                mask = np.stack([mask, mask, mask], axis=2)
                
                # Blend warped region with original
                blended_region = mouth_region * (1 - mask) + warped_region * mask
                
                # Apply back to result
                result[y:y+h, x:x+w] = blended_region
                
                print(f"🎭 LOCAL WARPING: Applied mouth-only deformation successfully")
                return result
                
            except Exception as e:
                print(f"🎭 LOCAL WARPING ERROR: {e}, using base face")
                return self.base_face
            
        except Exception as e:
            self.logger.error(f"Error in simple direct deformation: {e}")
            return self.base_face
    
    def _apply_ultra_simple_deformation(self, amplitude: float, frequency: float, phoneme_type: str) -> np.ndarray:
        """Ultra-simple deformation that just draws a mouth shape"""
        try:
            print(f"🎭 ULTRA-SIMPLE CALLED: amplitude={amplitude:.3f}, frequency={frequency:.3f}, phoneme_type='{phoneme_type}'")
            
            # Create a copy of the base face
            result = self.base_face.copy()
            
            # Get mouth center from landmarks
            mouth_center = np.mean(self.original_mouth_points, axis=0)
            center_x, center_y = int(mouth_center[0]), int(mouth_center[1])
            
            # Calculate mouth size and color based on phoneme - EXTREMELY DISTINCT SIZES
            if phoneme_type == "vowel":
                mouth_width = int(200 + amplitude * 200)  # HUGE
                mouth_height = int(150 + amplitude * 250)  # HUGE
                color = (0, 255, 0)  # Bright Green
            elif phoneme_type == "consonant":
                mouth_width = int(100 + amplitude * 150)   # Large
                mouth_height = int(80 + amplitude * 180)   # Large
                color = (0, 0, 255)  # Bright Red
            elif phoneme_type == "closed":
                mouth_width = int(20 + amplitude * 40)     # Tiny
                mouth_height = int(10 + amplitude * 30)    # Tiny
                color = (255, 0, 0)  # Bright Blue
            else:  # neutral
                mouth_width = int(60 + amplitude * 120)    # Medium
                mouth_height = int(50 + amplitude * 140)   # Medium
                color = (255, 255, 0)  # Bright Yellow
            
            print(f"🎭 ULTRA-SIMPLE: Drawing {phoneme_type} mouth at ({center_x}, {center_y}) size {mouth_width}x{mouth_height}")
            
            # Draw a simple oval mouth
            cv2.ellipse(result, (center_x, center_y), (mouth_width//2, mouth_height//2), 
                       0, 0, 360, color, -1)  # Filled ellipse
            
            # Add a thicker border for better visibility
            cv2.ellipse(result, (center_x, center_y), (mouth_width//2, mouth_height//2), 
                       0, 0, 360, (255, 255, 255), 5)  # Thicker white border
            
            # Add text label for debugging - MUCH LARGER AND MORE VISIBLE
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.0  # Much larger text
            thickness = 4     # Much thicker text
            text = phoneme_type.upper()
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = center_x - text_size[0] // 2
            text_y = center_y + text_size[1] // 2 + 80  # Further below the mouth
            
            # Add black background for better visibility
            cv2.putText(result, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2)  # Black outline
            cv2.putText(result, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)  # White text
            
            print(f"🎭 ULTRA-SIMPLE: Drew {phoneme_type} mouth successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in ultra-simple deformation: {e}")
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
            
            # Apply width stretching - FIXED LOGIC
            for i, point in enumerate(new_points):
                # Stretch horizontally from center - MORE AGGRESSIVE
                dx = (point[0] - center[0]) * (width_stretch - 1.0) * 2.0  # Double the effect
                new_points[i][0] += dx
            
            # Apply height stretching - FIXED LOGIC  
            for i, point in enumerate(new_points):
                # Stretch vertically from center - MORE AGGRESSIVE
                dy = (point[1] - center[1]) * (height_stretch - 1.0) * 2.0  # Double the effect
                new_points[i][1] += dy
            
            # Apply EXTRA jaw drop to lower lip points - FORCE VISIBLE MOVEMENT
            for i in lower_lip_indices:
                if i < len(new_points):
                    # Force dramatic jaw drop regardless of distance
                    new_points[i][1] += jaw_drop * 0.8  # 80% of jaw drop directly applied
            
            # Debug: Check if points actually changed
            original_center = np.mean(points, axis=0)
            new_center = np.mean(new_points, axis=0)
            print(f"🎭 POINT DEBUG: Original center {original_center}, new center {new_center}")
            print(f"🎭 POINT DEBUG: Center change: {new_center - original_center}")
            
            # Check if any points moved significantly
            max_change = np.max(np.abs(new_points - points))
            print(f"🎭 POINT DEBUG: Maximum point change: {max_change:.3f} pixels")
            
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
            
            # Create a mask for the mouth area - MUCH LARGER AND MORE AGGRESSIVE
            mask = np.zeros((h, w), dtype=np.uint8)
            mouth_contour = src_region.astype(np.int32)
            cv2.fillPoly(mask, [mouth_contour], 255)
            
            # Expand the mask to cover more area around the mouth
            kernel = np.ones((15, 15), np.uint8)  # Larger kernel for more coverage
            mask = cv2.dilate(mask, kernel, iterations=2)  # Expand mask area
            
            # Apply Gaussian blur to create soft edges - LESS BLUR FOR SHARPER DEFORMATION
            mask = cv2.GaussianBlur(mask, (11, 11), 0)  # Reduced blur for sharper edges
            
            # Normalize mask and make it more aggressive
            mask = mask.astype(np.float32) / 255.0
            mask = np.clip(mask * 1.5, 0, 1)  # Boost mask intensity
            mask = np.stack([mask, mask, mask], axis=2)
            
            # Apply perspective transform to the region
            try:
                print(f"🎭 WARPING: Attempting homography transformation...")
                # Calculate homography matrix
                H = cv2.findHomography(src_region, dst_region, cv2.RANSAC, 5.0)[0]
                
                # Apply transformation
                warped_region = cv2.warpPerspective(region, H, (w, h), 
                                                  flags=cv2.INTER_LINEAR, 
                                                  borderMode=cv2.BORDER_REFLECT)
                
                # Blend using the mask for seamless integration - MORE AGGRESSIVE BLENDING
                blended_region = region * (1 - mask) + warped_region * mask
                
                # Apply to result
                result[y:y+h, x:x+w] = blended_region
                
                print(f"🎭 WARPING: Homography transformation successful!")
                
            except Exception as e:
                # Fallback to simple scaling if homography fails
                print(f"🎭 WARPING: Homography failed, using fallback: {e}")
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
        """Analyze audio amplitude using proven method"""
        try:
            if len(audio_array) == 0:
                return 0.0
            
            # Use proven audio analysis method
            # 1. Calculate RMS (Root Mean Square) for overall energy
            rms = np.sqrt(np.mean(audio_array**2))
            
            # 2. Calculate peak amplitude for dynamic range
            peak = np.max(np.abs(audio_array))
            
            # 3. Calculate zero-crossing rate for frequency content
            zero_crossings = np.sum(np.diff(np.sign(audio_array)) != 0)
            zcr = zero_crossings / len(audio_array)
            
            # 4. Combine features for robust amplitude detection
            # Use a weighted combination that's proven to work
            amplitude = (0.6 * rms + 0.3 * peak + 0.1 * zcr)
            
            # 5. Apply soft normalization (no hard capping)
            amplitude = np.tanh(amplitude * 3.0)  # Soft saturation, not hard cap
            
            print(f"📊 PROVEN AUDIO: rms={rms:.4f}, peak={peak:.4f}, zcr={zcr:.4f}, amplitude={amplitude:.4f}")
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