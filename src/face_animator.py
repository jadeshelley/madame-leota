"""
Face Animator - Clean and Simple
Uses only the clean mouth animator that works reliably on Pi
No complex dependencies - just basic OpenCV operations
"""

import cv2
import numpy as np
import logging
import asyncio
import time
import random
import math
from pathlib import Path
from typing import List, Dict, Optional
from config import *

class FaceAnimator:
    """Clean face animator that works reliably on Pi"""
    
    def __init__(self, display_manager):
        self.display_manager = display_manager
        self.logger = logging.getLogger(__name__)
        
        # Animation state
        self.is_speaking = False
        self.current_state = "idle"
        self.current_face = None
        self.idle_animation_running = False
        
        # Initialize animation system attributes
        self.real_mouth_manipulator = None
        self.clean_mouth_animator = None
        
        # Initialize real mouth manipulator (actually manipulates images)
        try:
            print("🎭 DEBUG: Attempting to import real mouth manipulator...")
            from src.real_mouth_manipulator import RealMouthManipulator
            self.real_mouth_manipulator = RealMouthManipulator()
            print("✅ REAL MOUTH: Real mouth manipulation system initialized")
            self.logger.info("✅ Real mouth manipulation system initialized")
            
            # Load base face for real mouth manipulation
            try:
                base_face_path = Path(FACE_ASSETS_DIR) / "mouth_closed.png"
                if base_face_path.exists():
                    print(f"🎭 REAL MOUTH: Loading base face from {base_face_path}")
                    success = self.real_mouth_manipulator.load_base_face(str(base_face_path))
                    if success:
                        print("✅ REAL MOUTH: Base face loaded successfully")
                        self.logger.info("✅ Real mouth base face loaded successfully")
                    else:
                        print("❌ REAL MOUTH: Failed to load base face")
                        self.logger.error("❌ Real mouth failed to load base face")
                        self.real_mouth_manipulator = None
                else:
                    print(f"❌ REAL MOUTH: Base face not found at {base_face_path}")
                    self.logger.error(f"❌ Real mouth base face not found at {base_face_path}")
                    self.real_mouth_manipulator = None
            except Exception as e:
                print(f"❌ REAL MOUTH: Error loading base face: {e}")
                self.logger.error(f"❌ Real mouth error loading base face: {e}")
                self.real_mouth_manipulator = None
                
        except Exception as e:
            print(f"❌ REAL MOUTH: Failed to initialize: {e}")
            self.logger.warning(f"Real mouth initialization failed: {e}")
            self.real_mouth_manipulator = None
        
        # Fallback to clean mouth animator if real manipulator fails
        if not self.real_mouth_manipulator:
            try:
                print("🎭 DEBUG: Attempting to import clean mouth animator as fallback...")
                from src.clean_mouth_animator import CleanMouthAnimator
                self.clean_mouth_animator = CleanMouthAnimator()
                print("✅ CLEAN MOUTH: Clean mouth animation system initialized (fallback)")
                self.logger.info("✅ Clean mouth animation system initialized (fallback)")
                
                # Load mouth shapes for clean mouth system
                try:
                    faces_dir = FACE_ASSETS_DIR
                    print(f"🎭 CLEAN MOUTH: Loading mouth shapes from {faces_dir}")
                    success = self.clean_mouth_animator.load_mouth_shapes(faces_dir)
                    if success:
                        print("✅ CLEAN MOUTH: Mouth shapes loaded successfully")
                        self.logger.info("✅ Clean mouth shapes loaded successfully")
                    else:
                        print("❌ CLEAN MOUTH: Failed to load mouth shapes")
                        self.logger.error("❌ Clean mouth failed to load mouth shapes")
                        self.clean_mouth_animator = None
                except Exception as e:
                    print(f"❌ CLEAN MOUTH: Error loading mouth shapes: {e}")
                    self.logger.error(f"❌ Clean mouth error loading mouth shapes: {e}")
                    self.clean_mouth_animator = None
                    
            except Exception as e:
                print(f"❌ CLEAN MOUTH: Failed to initialize: {e}")
                self.logger.warning(f"Clean mouth initialization failed: {e}")
                self.clean_mouth_animator = None
        
        # Initialize face_images attribute
        self.face_images = {}
        
        # Load face assets
        self._load_face_assets()
        
        print(f"🎬 ANIMATOR: Animation system ready - real mouth: {self.real_mouth_manipulator is not None}, clean mouth: {self.clean_mouth_animator is not None}")
        self.logger.info(f"Face animator initialized with real mouth: {self.real_mouth_manipulator is not None}, clean mouth: {self.clean_mouth_animator is not None}")
        
        # Animation timing
        self.animation_start_time = 0
        self.current_frame = 0
        
        # Mouth shape cache
        self.mouth_shapes = {}
        
        # Current face for smooth morphing
        self._current_face = None
    
    def _load_face_assets(self):
        """Load face images for animation"""
        try:
            faces_dir = Path(FACE_ASSETS_DIR)
            if not faces_dir.exists():
                self.logger.warning(f"Face assets directory not found: {faces_dir}")
                return
            
            # Load face images
            for face_file in faces_dir.glob("*.png"):
                face_name = face_file.stem
                face_image = cv2.imread(str(face_file))
                if face_image is not None:
                    self.face_images[face_name] = face_image
                    self.logger.info(f"Loaded face image: {face_name}")
                else:
                    self.logger.warning(f"Failed to load face image: {face_file}")
            
            # Set current face
            if self.face_images:
                self.current_face = self.face_images.get('mouth_closed', list(self.face_images.values())[0])
                self.logger.info(f"Loaded {len(self.face_images)} face images")
            else:
                self.logger.warning("No face images loaded")
                
        except Exception as e:
            self.logger.error(f"❌ ERROR: Failed to load face assets: {e}")
    
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
        """Animate speaking using real mouth manipulator or clean mouth animator"""
        try:
            self.logger.info("animate_speaking_with_audio called")
            
            # Use real mouth manipulator if available (actually manipulates images)
            if self.real_mouth_manipulator:
                await self._animate_with_real_mouth(audio_data, phonemes)
                return
            
            # Fallback to clean mouth animator if available
            elif self.clean_mouth_animator:
                await self._animate_with_clean_mouth(audio_data, phonemes)
                return
            
            # Final fallback to simple phoneme animation
            else:
                self.logger.warning("No animation systems available, falling back to phoneme animation")
                await self.animate_speaking(phonemes)
                return
        except Exception as e:
            self.logger.error(f"Speaking animation error: {e}")
            self.is_speaking = False
    
    async def _animate_with_clean_mouth(self, audio_data: bytes, phonemes: List[dict]):
        """Animate using clean mouth animator"""
        try:
            self.is_speaking = True
            self.current_state = "speaking"
            
            # Calculate total duration and frame rate
            total_duration = sum(p.get('duration', 0) for p in phonemes) / 1000.0
            frame_rate = 15  # 15 FPS for smooth animation
            frame_duration = 1.0 / frame_rate
            total_frames = int(total_duration * frame_rate)
            
            self.logger.info(f"Clean mouth animation: {total_duration:.2f}s, {frame_rate} FPS, {total_frames} frames")
            
            # Convert full audio once
            audio_array = self.clean_mouth_animator._bytes_to_audio_array(audio_data)
            if len(audio_array) == 0:
                self.logger.error("Failed to convert audio, falling back to phoneme animation")
                await self.animate_speaking(phonemes)
                return
            
            # Calculate samples per frame
            samples_per_frame = len(audio_array) // total_frames if total_frames > 0 else len(audio_array)
            
            # Animation loop
            for frame in range(total_frames):
                # Extract audio chunk for this frame
                start_idx = frame * samples_per_frame
                end_idx = min(start_idx + samples_per_frame, len(audio_array))
                audio_chunk = audio_array[start_idx:end_idx]
                
                # Handle empty audio chunks
                if len(audio_chunk) == 0:
                    print(f"⚠️ AUDIO CHUNK EMPTY: Frame {frame}, using last available audio")
                    if frame > 0:
                        start_idx = max(0, len(audio_array) - samples_per_frame)
                        audio_chunk = audio_array[start_idx:]
                    else:
                        audio_chunk = audio_array[:min(1024, len(audio_array))]
                
                # Generate face using clean mouth system
                face = self.clean_mouth_animator.generate_face_for_audio_chunk(audio_chunk)
                
                # Debug output every 5 frames
                if frame % 5 == 0:
                    print(f"🎭 ANIMATION DEBUG: Frame {frame}, audio chunk: {len(audio_chunk)} samples, face shape: {face.shape}")
                
                # Display the face
                try:
                    print(f"🖥️ DISPLAY DEBUG: Frame {frame} - About to display face...")
                    
                    # Scale face to fit screen
                    target_height = 600
                    aspect_ratio = face.shape[1] / face.shape[0]
                    target_width = int(target_height * aspect_ratio)
                    
                    scaled_face = cv2.resize(face, (target_width, target_height))
                    
                    # Calculate screen position to center the face
                    screen_width, screen_height = self.display_manager.get_screen_size()
                    screen_center_x = (screen_width - target_width) // 2
                    screen_center_y = (screen_height - target_height) // 2
                    screen_position = (screen_center_x, screen_center_y)
                    
                    # Clear screen and display
                    self.display_manager.clear_screen()
                    self.display_manager.display_image(scaled_face, screen_position)
                    
                    print(f"✅ DISPLAY DEBUG: Frame {frame} - Face displayed successfully")
                    
                except Exception as e:
                    print(f"❌ DISPLAY ERROR: Frame {frame} - {e}")
                    # Fallback to simple display
                    self.display_manager.clear_screen()
                    self.display_manager.display_image(face, (0, 0))
                
                await asyncio.sleep(frame_duration)
                
        except Exception as e:
            self.logger.error(f"Clean mouth animation error: {e}")
            await self.animate_speaking(phonemes)
        finally:
            self.is_speaking = False
    
    async def _animate_with_real_mouth(self, audio_data: bytes, phonemes: List[dict]):
        """Animate using real mouth manipulator (actually manipulates images)"""
        try:
            self.is_speaking = True
            self.current_state = "speaking"
            
            # Calculate total duration and frame rate
            total_duration = sum(p.get('duration', 0) for p in phonemes) / 1000.0
            frame_rate = 15  # 15 FPS for smooth animation
            frame_duration = 1.0 / frame_rate
            total_frames = int(total_duration * frame_rate)
            
            self.logger.info(f"Real mouth animation: {total_duration:.2f}s, {frame_rate} FPS, {total_frames} frames")
            
            # Convert full audio once
            audio_array = self.real_mouth_manipulator._bytes_to_audio_array(audio_data)
            if len(audio_array) == 0:
                self.logger.error("Failed to convert audio, falling back to phoneme animation")
                await self.animate_speaking(phonemes)
                return
            
            # Calculate samples per frame
            samples_per_frame = len(audio_array) // total_frames if total_frames > 0 else len(audio_array)
            
            # Animation loop
            for frame in range(total_frames):
                # Extract audio chunk for this frame
                start_idx = frame * samples_per_frame
                end_idx = min(start_idx + samples_per_frame, len(audio_array))
                audio_chunk = audio_array[start_idx:end_idx]
                
                # Handle empty audio chunks
                if len(audio_chunk) == 0:
                    print(f"⚠️ AUDIO CHUNK EMPTY: Frame {frame}, using last available audio")
                    if frame > 0:
                        start_idx = max(0, len(audio_array) - samples_per_frame)
                        audio_chunk = audio_array[start_idx:]
                    else:
                        audio_chunk = audio_array[:min(1024, len(audio_array))]
                
                # Generate face using real mouth manipulation
                face = self.real_mouth_manipulator.generate_face_for_audio_chunk(audio_chunk)
                
                # Debug output every 5 frames
                if frame % 5 == 0:
                    print(f"🎭 ANIMATION DEBUG: Frame {frame}, audio chunk: {len(audio_chunk)} samples, face shape: {face.shape}")
                
                # Display the face
                try:
                    print(f"🖥️ DISPLAY DEBUG: Frame {frame} - About to display face...")
                    
                    # Scale face to fit screen
                    target_height = 600
                    aspect_ratio = face.shape[1] / face.shape[0]
                    target_width = int(target_height * aspect_ratio)
                    
                    scaled_face = cv2.resize(face, (target_width, target_height))
                    
                    # Calculate screen position to center the face
                    screen_width, screen_height = self.display_manager.get_screen_size()
                    screen_center_x = (screen_width - target_width) // 2
                    screen_center_y = (screen_height - target_height) // 2
                    screen_position = (screen_center_x, screen_center_y)
                    
                    # Clear screen and display
                    self.display_manager.clear_screen()
                    self.display_manager.display_image(scaled_face, screen_position)
                    
                    print(f"✅ DISPLAY DEBUG: Frame {frame} - Face displayed successfully")
                    
                except Exception as e:
                    print(f"❌ DISPLAY ERROR: Frame {frame} - {e}")
                    # Fallback to simple display
                    self.display_manager.clear_screen()
                    self.display_manager.display_image(face, (0, 0))
                
                await asyncio.sleep(frame_duration)
                
        except Exception as e:
            self.logger.error(f"Real mouth animation error: {e}")
            await self.animate_speaking(phonemes)
        finally:
            self.is_speaking = False
    
    async def _display_mouth_shape(self, mouth_shape: str, duration: float):
        """Display mouth shape using simple morphing"""
        try:
            target_image = self.face_images.get(mouth_shape, self.face_images.get('mouth_closed'))
            
            if target_image is not None:
                # Simple morphing - just display the target shape
                self.display_manager.clear_screen()
                self.display_manager.display_image(target_image, (0, 0))
                self.display_manager.update_display()
                
                # Hold for duration
                if duration > 0:
                    await asyncio.sleep(duration)
                
                # Store current face for next transition
                self._current_face = target_image.copy()
            
        except Exception as e:
            self.logger.error(f"Error displaying mouth shape {mouth_shape}: {e}")
    
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
            
            if self.real_mouth_manipulator:
                self.real_mouth_manipulator.cleanup()
            if self.clean_mouth_animator:
                self.clean_mouth_animator.cleanup()
            
            self.logger.info("Face Animator cleaned up")
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}") 