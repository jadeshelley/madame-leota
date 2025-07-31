"""
Face Animator for Madame Leota
Controls facial expressions, lip sync, and idle animations
Supports both smooth morphing and real-time face manipulation
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

# Import real-time face manipulator if enabled
if USE_REALTIME_FACE_MANIPULATION:
    try:
        from .realtime_face_manipulator import RealtimeFaceManipulator
        REALTIME_AVAILABLE = True
    except ImportError as e:
        logging.warning(f"Real-time face manipulation not available: {e}")
        REALTIME_AVAILABLE = False
else:
    REALTIME_AVAILABLE = False

# Import audio-driven face system if enabled
if USE_AUDIO_DRIVEN_FACE:
    try:
        logging.info("Attempting to import AudioDrivenFace...")
        from .audio_driven_face import AudioDrivenFace
        AUDIO_DRIVEN_AVAILABLE = True
        logging.info("✅ AudioDrivenFace imported successfully!")
    except ImportError as e:
        logging.warning(f"❌ Audio-driven face manipulation not available: {e}")
        AUDIO_DRIVEN_AVAILABLE = False
else:
    logging.info("USE_AUDIO_DRIVEN_FACE is False in config")
    AUDIO_DRIVEN_AVAILABLE = False

class FaceAnimator:
    def __init__(self, display_manager):
        self.logger = logging.getLogger(__name__)
        self.display_manager = display_manager
        
        # Log configuration status
        self.logger.info(f"Face Animator Config - USE_AUDIO_DRIVEN_FACE: {USE_AUDIO_DRIVEN_FACE}")
        self.logger.info(f"Face Animator Config - AUDIO_DRIVEN_AVAILABLE: {AUDIO_DRIVEN_AVAILABLE}")
        
        # Animation state
        self.current_state = "idle"
        self.is_speaking = False
        self.idle_animation_running = False
        
        # Load face assets
        print("📁 DEBUG: About to load face assets...")
        self.face_images = self._load_face_assets()
        print("✅ DEBUG: Face assets loaded!")
        
        # Animation timing
        self.animation_start_time = 0
        self.current_frame = 0
        
        # Mouth shape cache
        self.mouth_shapes = {}
        
        # Current face for smooth morphing
        self._current_face = None
        
        # Initialize real-time face manipulator if available
        self.realtime_manipulator = None
        if USE_REALTIME_FACE_MANIPULATION and REALTIME_AVAILABLE:
            try:
                self.realtime_manipulator = RealtimeFaceManipulator()
                # Load base face for manipulation
                base_face_path = Path(FACE_ASSETS_DIR) / "mouth_closed.png"
                if base_face_path.exists():
                    success = self.realtime_manipulator.load_base_face(str(base_face_path))
                    if success:
                        self.logger.info("Real-time face manipulation enabled")
                    else:
                        self.logger.warning("Failed to load base face for manipulation, falling back to morphing")
                        self.realtime_manipulator = None
                else:
                    self.logger.warning("No base face found for manipulation, falling back to morphing")
                    self.realtime_manipulator = None
            except Exception as e:
                self.logger.warning(f"Real-time face manipulation setup failed: {e}, falling back to morphing")
                self.realtime_manipulator = None
        
        # Initialize audio-driven face system if available (highest priority)
        self.audio_driven_face = None
        self.logger.info(f"🔍 Checking audio-driven face: USE_AUDIO_DRIVEN_FACE={USE_AUDIO_DRIVEN_FACE}, AVAILABLE={AUDIO_DRIVEN_AVAILABLE}")
        
        if USE_AUDIO_DRIVEN_FACE and AUDIO_DRIVEN_AVAILABLE:
            try:
                print("🚀 DEBUG: About to create AudioDrivenFace instance...")
                self.logger.info("🚀 Initializing AudioDrivenFace...")
                self.audio_driven_face = AudioDrivenFace()
                print("✅ DEBUG: AudioDrivenFace instance created!")
                self.logger.info("✅ AudioDrivenFace created successfully")
                
                # Load base face for manipulation
                base_face_path = Path(FACE_ASSETS_DIR) / "mouth_closed.png"
                self.logger.info(f"📁 Looking for base face at: {base_face_path}")
                
                if base_face_path.exists():
                    print("✅ DEBUG: Base face file found, about to load...")
                    self.logger.info("✅ Base face file found, loading...")
                    success = self.audio_driven_face.load_base_face(str(base_face_path))
                    print("✅ DEBUG: Base face loading completed!")
                    if success:
                        self.logger.info("🎭 Audio-driven deepfake-like face manipulation enabled!")
                    else:
                        self.logger.error("❌ Failed to load base face for audio-driven manipulation")
                        self.audio_driven_face = None
                else:
                    self.logger.error(f"❌ No base face found at {base_face_path}")
                    self.audio_driven_face = None
            except Exception as e:
                self.logger.error(f"❌ Audio-driven face setup failed: {e}")
                self.logger.error(f"Exception type: {type(e).__name__}")
                import traceback
                self.logger.error(f"Full traceback: {traceback.format_exc()}")
                self.audio_driven_face = None
        else:
            self.logger.warning("⚠️  Audio-driven face not available - will use fallback animation")
        
        # Create base face if no assets found
        print("🔍 DEBUG: Checking if face assets need to be created...")
        if not self.face_images:
            print("⚠️ DEBUG: No face assets found, creating default...")
            self._create_default_face()
        
        # Initialize current face
        print("🎭 DEBUG: Setting up current face...")
        self._current_face = self.face_images.get('mouth_closed', self.face_images.get('base'))
        print("✅ DEBUG: Current face initialized!")
        
        animation_type = "audio-driven deepfake-like" if self.audio_driven_face else \
                         "real-time manipulation" if self.realtime_manipulator else \
                         "enhanced morphing"
        print(f"🎬 DEBUG: About to finish FaceAnimator init with: {animation_type}")
        self.logger.info(f"🎬 Face Animator initialized with: {animation_type}")
        print("🎯 DEBUG: FaceAnimator.__init__() completed successfully!")
    
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
        """Create a more realistic default mystical face"""
        try:
            # Face dimensions 
            width, height = 400, 500
            
            # Create face with better colors
            face = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Mystical background glow
            center = (width // 2, height // 2)
            for i in range(5):
                glow_radius = 180 - i * 30
                glow_intensity = 20 - i * 3
                cv2.circle(face, center, glow_radius, (glow_intensity, glow_intensity//2, glow_intensity//3), -1)
            
            # Face shape (more realistic oval)
            face_color = (180, 150, 120)  # Warmer skin tone
            cv2.ellipse(face, (width//2, height//2 + 20), (120, 160), 0, 0, 360, face_color, -1)
            
            # Cheekbones and face contour
            cheek_color = (160, 130, 100)
            cv2.ellipse(face, (width//2 - 60, height//2), (40, 80), 15, 0, 360, cheek_color, -1)
            cv2.ellipse(face, (width//2 + 60, height//2), (40, 80), -15, 0, 360, cheek_color, -1)
            
            # Eyes (more mystical)
            eye_color = (80, 40, 120)  # Purple-ish mystical eyes
            pupil_color = (200, 180, 255)  # Glowing pupils
            
            # Left eye
            cv2.ellipse(face, (width//2 - 40, height//2 - 30), (25, 15), 0, 0, 360, eye_color, -1)
            cv2.circle(face, (width//2 - 40, height//2 - 30), 8, pupil_color, -1)
            cv2.circle(face, (width//2 - 40, height//2 - 30), 3, (255, 255, 255), -1)
            
            # Right eye  
            cv2.ellipse(face, (width//2 + 40, height//2 - 30), (25, 15), 0, 0, 360, eye_color, -1)
            cv2.circle(face, (width//2 + 40, height//2 - 30), 8, pupil_color, -1)
            cv2.circle(face, (width//2 + 40, height//2 - 30), 3, (255, 255, 255), -1)
            
            # Eyebrows
            brow_color = (100, 80, 60)
            cv2.ellipse(face, (width//2 - 40, height//2 - 50), (30, 8), 15, 0, 180, brow_color, -1)
            cv2.ellipse(face, (width//2 + 40, height//2 - 50), (30, 8), 165, 0, 180, brow_color, -1)
            
            # Nose (more refined)
            nose_color = (160, 130, 100)
            nose_points = np.array([
                [width//2, height//2 - 10],
                [width//2 - 8, height//2 + 10],
                [width//2 + 8, height//2 + 10]
            ], np.int32)
            cv2.fillPoly(face, [nose_points], nose_color)
            
            # Create different mouth shapes
            self._create_mouth_shapes(face, width, height)
            
            # Store base face
            self.face_images['base'] = face.copy()
            
            self.logger.info("Created improved mystical default face")
            
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
    
    async def animate_speaking_with_audio(self, audio_data: bytes, phonemes: List[dict]):
        """Animate speaking using audio-driven face manipulation for deepfake-like results"""
        try:
            self.logger.info("animate_speaking_with_audio called")
            
            if not self.audio_driven_face:
                self.logger.warning("No audio_driven_face available, falling back to phoneme animation")
                await self.animate_speaking(phonemes)
                return
            
            self.logger.info(f"Using audio-driven face manipulation with {len(audio_data)} bytes of audio")
            
            self.is_speaking = True
            self.current_state = "speaking"
            
            # Calculate total duration and frame rate
            total_duration = sum(p.get('duration', 0) for p in phonemes) / 1000.0
            frame_rate = 15  # 15 FPS for smooth animation
            frame_duration = 1.0 / frame_rate
            total_frames = int(total_duration * frame_rate)
            
            self.logger.info(f"Real-time animation: {total_duration:.2f}s, {frame_rate} FPS, {total_frames} frames")
            
            # Convert full audio once
            audio_array = self.audio_driven_face._bytes_to_audio_array(audio_data)
            if len(audio_array) == 0:
                self.logger.error("Failed to convert audio, falling back to phoneme animation")
                await self.animate_speaking(phonemes)
                return
            
            # Calculate samples per frame
            samples_per_frame = len(audio_array) // total_frames if total_frames > 0 else len(audio_array)
            self.logger.info(f"Processing {samples_per_frame} samples per frame")
            
            # Real-time animation loop
            for frame in range(total_frames):
                if not self.is_speaking:  # Allow interruption
                    break
                
                frame_start_time = time.time()
                
                # Extract audio chunk for this frame
                start_sample = frame * samples_per_frame
                end_sample = min((frame + 1) * samples_per_frame, len(audio_array))
                audio_chunk = audio_array[start_sample:end_sample]
                
                # Generate face for this audio chunk
                if len(audio_chunk) > 0:
                    deepfake_face = await self._generate_face_for_chunk(audio_chunk, frame_duration)
                    
                    # Display the frame
                    self.display_manager.clear_screen()
                    self.display_manager.display_face(deepfake_face)
                    self.display_manager.update_display()
                
                # Maintain frame rate
                elapsed = time.time() - frame_start_time
                sleep_time = max(0, frame_duration - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            # Return to idle
            self.is_speaking = False
            self.current_state = "idle"
            
        except Exception as e:
            self.logger.error(f"Error in audio-driven speaking animation: {e}")
            self.logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
            # Fallback to regular animation
            await self.animate_speaking(phonemes)
    
    async def _generate_face_for_chunk(self, audio_chunk: np.ndarray, duration: float) -> np.ndarray:
        """Generate face for a single audio chunk"""
        try:
            if self.audio_driven_face.base_face is None:
                return self.audio_driven_face.base_face
            
            # Analyze this chunk
            amplitude = self.audio_driven_face._analyze_amplitude_simple(audio_chunk)
            dominant_freq = self.audio_driven_face._analyze_frequency_simple(audio_chunk)
            speech_energy = self.audio_driven_face._analyze_speech_energy(audio_chunk)
            
            # Smooth features over time
            smoothed_amplitude = self.audio_driven_face._smooth_feature(amplitude, self.audio_driven_face.amplitude_history)
            smoothed_frequency = self.audio_driven_face._smooth_feature(dominant_freq, self.audio_driven_face.frequency_history)
            
            # Map to facial parameters
            jaw_drop = self.audio_driven_face._map_amplitude_to_jaw(smoothed_amplitude)
            lip_width = self.audio_driven_face._map_frequency_to_lip_width(smoothed_frequency)
            lip_height = self.audio_driven_face._map_amplitude_to_lip_height(smoothed_amplitude)
            micro_movement = self.audio_driven_face._generate_micro_movement(speech_energy)
            
            # Generate face with these parameters
            deepfake_face = self.audio_driven_face._apply_deepfake_deformation(
                self.audio_driven_face.base_face.copy(),
                jaw_drop, lip_width, lip_height, micro_movement
            )
            
            return deepfake_face
            
        except Exception as e:
            self.logger.error(f"Error generating face for chunk: {e}")
            return self.audio_driven_face.base_face if self.audio_driven_face else self.face_images.get('mouth_closed')
    
    async def _display_mouth_shape(self, mouth_shape: str, duration: float):
        """Display mouth shape using real-time manipulation or smooth morphing"""
        try:
            if self.realtime_manipulator:
                # Use real-time face manipulation (Option 4)
                animated_face = await self.realtime_manipulator.animate_phoneme(mouth_shape, duration)
                
                # Display properly
                self.display_manager.clear_screen()
                self.display_manager.display_face(animated_face)
                self.display_manager.update_display()
                
                # Hold for duration
                if duration > 0:
                    await asyncio.sleep(duration)
            else:
                # Fallback to smooth morphing (Option 1)
                await self._display_mouth_shape_morphing(mouth_shape, duration)
            
        except Exception as e:
            self.logger.error(f"Error displaying mouth shape {mouth_shape}: {e}")
            # Emergency fallback
            await self._display_mouth_shape_morphing(mouth_shape, duration)
    
    async def _display_mouth_shape_morphing(self, mouth_shape: str, duration: float):
        """Enhanced smooth morphing method with dynamic timing"""
        try:
            target_image = self.face_images.get(mouth_shape, self.face_images.get('mouth_closed'))
            
            if target_image is not None:
                # Get current face for smooth transition
                current_image = getattr(self, '_current_face', self.face_images.get('mouth_closed'))
                
                # Enhanced morphing settings
                if USE_ENHANCED_MORPHING:
                    morph_steps = MORPH_STEPS
                    # Dynamic timing - faster start, slower end for more natural feel
                    morph_duration = min(0.15, duration / 2)  # Slightly longer for smoothness
                else:
                    morph_steps = 5
                    morph_duration = min(0.1, duration / 2)
                
                step_time = morph_duration / morph_steps
                
                for step in range(morph_steps + 1):
                    # Dynamic easing for more natural movement
                    if USE_ENHANCED_MORPHING and MORPH_TIMING == "dynamic":
                        # Ease-in-out curve (slow-fast-slow)
                        t = step / morph_steps
                        alpha = t * t * (3.0 - 2.0 * t)  # Smooth step function
                    else:
                        alpha = step / morph_steps  # Linear
                    
                    # Blend images for smooth transition
                    blended = cv2.addWeighted(current_image, 1 - alpha, target_image, alpha, 0)
                    
                    # Add subtle brightness variation for more life-like effect
                    if USE_ENHANCED_MORPHING:
                        # Slight brightness pulse during speech
                        brightness_factor = 1.0 + 0.05 * np.sin(step * 2)
                        blended = cv2.convertScaleAbs(blended, alpha=brightness_factor, beta=0)
                    
                    # Display properly
                    self.display_manager.clear_screen()
                    self.display_manager.display_face(blended)
                    self.display_manager.update_display()
                    
                    if step < morph_steps:
                        await asyncio.sleep(step_time)
                
                # Hold the target shape for remaining duration
                hold_time = duration - morph_duration
                if hold_time > 0:
                    # Add subtle animation during hold for more realistic effect
                    if USE_ENHANCED_MORPHING and hold_time > 0.3:
                        await self._animate_hold_phase(target_image, hold_time)
                    else:
                        await asyncio.sleep(hold_time)
                
                # Store current face for next transition
                self._current_face = target_image.copy()
            
        except Exception as e:
            self.logger.error(f"Error in enhanced morphing for {mouth_shape}: {e}")
    
    async def _animate_hold_phase(self, face_image: np.ndarray, duration: float):
        """Add subtle animation during the hold phase for more realism"""
        try:
            steps = max(3, int(duration * 10))  # ~10 FPS during hold
            step_time = duration / steps
            
            for i in range(steps):
                # Subtle micro-movements
                offset_x = int(2 * np.sin(i * 0.5))  # Small horizontal sway
                offset_y = int(1 * np.cos(i * 0.3))  # Tiny vertical movement
                
                # Apply micro-movement
                h, w = face_image.shape[:2]
                M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
                animated = cv2.warpAffine(face_image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                
                # Display properly
                self.display_manager.clear_screen()
                self.display_manager.display_face(animated)
                self.display_manager.update_display()
                
                await asyncio.sleep(step_time)
                
        except Exception as e:
            self.logger.error(f"Error in hold phase animation: {e}")
            # Fallback to static hold
            await asyncio.sleep(duration)
    
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
    
    def _scale_face(self, face_image: np.ndarray) -> np.ndarray:
        """Scale face image to display dimensions"""
        try:
            target_height = int(PROJECTOR_HEIGHT * FACE_SCALE)
            target_width = int(PROJECTOR_WIDTH * FACE_SCALE)
            
            # Maintain aspect ratio
            h, w = face_image.shape[:2]
            aspect_ratio = w / h
            
            if target_width / target_height > aspect_ratio:
                # Height is limiting factor
                new_height = target_height
                new_width = int(target_height * aspect_ratio)
            else:
                # Width is limiting factor
                new_width = target_width
                new_height = int(target_width / aspect_ratio)
            
            # Resize image
            scaled = cv2.resize(face_image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            
            return scaled
            
        except Exception as e:
            self.logger.error(f"Error scaling face: {e}")
            return face_image
    
    def cleanup(self):
        """Cleanup animator resources"""
        try:
            self.is_speaking = False
            self.idle_animation_running = False
            
            if self.realtime_manipulator:
                self.realtime_manipulator.cleanup()
            
            self.logger.info("Face Animator cleaned up")
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

# Import random for idle animation
import random 