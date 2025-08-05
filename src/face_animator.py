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

# Import Wav2Lip AI system if enabled
if USE_WAV2LIP:
    try:
        logging.info("Attempting to import Wav2Lip AI...")
        from .wav2lip_animator import get_wav2lip_animator
        WAV2LIP_AVAILABLE = True
        logging.info("✅ Wav2Lip AI imported successfully!")
    except ImportError as e:
        logging.warning(f"❌ Wav2Lip AI not available: {e}")
        WAV2LIP_AVAILABLE = False
else:
    logging.info("USE_WAV2LIP is False in config")
    WAV2LIP_AVAILABLE = False

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
        
        # Initialize different animation systems based on config
        self.wav2lip_animator = None
        self.dlib_face_animator = None
        self.audio_driven_face = None
        self.current_face = None
        
        # Try to initialize Wav2Lip AI first (highest quality)
        if USE_WAV2LIP and WAV2LIP_AVAILABLE:
            try:
                print("🤖 DEBUG: Attempting to initialize Wav2Lip AI...")
                self.wav2lip_animator = get_wav2lip_animator()
                print("✅ WAV2LIP: AI lip sync system initialized")
                self.logger.info("✅ Wav2Lip AI system initialized")
            except Exception as e:
                print(f"⚠️ WAV2LIP: Failed to initialize: {e}")
                self.logger.warning(f"Wav2Lip initialization failed: {e}")
                self.wav2lip_animator = None
        
        # Fallback to dlib facial landmarks if Wav2Lip not available
        if not self.wav2lip_animator:
            try:
                print("🔍 DEBUG: Attempting to import dlib system...")
                from src.dlib_face_animator import DlibFaceAnimator
                print("✅ DEBUG: dlib import successful, creating instance...")
                self.dlib_face_animator = DlibFaceAnimator()
                print("✅ DLIB: Facial landmark system initialized")
                self.logger.info("✅ dlib facial landmark system initialized")
                
                # Load base face for dlib system
                try:
                    base_face_path = Path(FACE_ASSETS_DIR) / "realistic_face.jpg"
                    if base_face_path.exists():
                        print(f"🎭 DLIB: Loading base face from {base_face_path}")
                        success = self.dlib_face_animator.load_base_face(str(base_face_path))
                        if success:
                            print("✅ DLIB: Base face loaded with facial landmarks detected")
                            self.logger.info("✅ dlib base face loaded successfully")
                        else:
                            print("❌ DLIB: Failed to load base face")
                            self.logger.error("❌ dlib failed to load base face")
                            self.dlib_face_animator = None
                    else:
                        print(f"❌ DLIB: Base face not found at {base_face_path}")
                        self.logger.error(f"❌ dlib base face not found at {base_face_path}")
                        self.dlib_face_animator = None
                except Exception as e:
                    print(f"❌ DLIB: Error loading base face: {e}")
                    self.logger.error(f"❌ dlib error loading base face: {e}")
                    self.dlib_face_animator = None
                    
            except ImportError as ie:
                print(f"⚠️ DLIB: Import failed - {ie}")
                print("⚠️ DLIB: Likely missing dlib package - run: pip install dlib")
                self.logger.warning(f"dlib import failed: {ie}")
                self.dlib_face_animator = None
            except Exception as e:
                print(f"⚠️ DLIB: Failed to initialize dlib system: {e}")
                print(f"⚠️ DLIB: Error type: {type(e).__name__}")
                import traceback
                print(f"⚠️ DLIB: Traceback: {traceback.format_exc()}")
                self.logger.warning(f"Could not initialize dlib system: {e}")
                self.dlib_face_animator = None
        
        # Final fallback to audio-driven system
        if not self.wav2lip_animator and not self.dlib_face_animator:
            if USE_AUDIO_DRIVEN_FACE:
                try:
                    from src.audio_driven_face import AudioDrivenFace
                    self.audio_driven_face = AudioDrivenFace()
                    print("✅ FALLBACK: Audio-driven face system initialized")
                    self.logger.info("✅ Audio-driven face system initialized")
                except Exception as e:
                    print(f"⚠️ FALLBACK: Audio-driven initialization failed: {e}")
                    self.logger.warning(f"Audio-driven face initialization failed: {e}")
        
        # Initialize face_images attribute
        self.face_images = {}
        
        # Load face assets
        self._load_face_assets()
        
        print(f"🎬 ANIMATOR: Animation system ready - wav2lip: {self.wav2lip_animator is not None}, dlib: {self.dlib_face_animator is not None}, audio-driven: {self.audio_driven_face is not None}")
        self.logger.info(f"Face animator initialized with systems - wav2lip: {self.wav2lip_animator is not None}, dlib: {self.dlib_face_animator is not None}, audio-driven: {self.audio_driven_face is not None}")
        
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
                from .audio_driven_face import AudioDrivenFace
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
    
    def _load_face_assets(self):
        """Load face images for animation"""
        try:
            # Load base face for dlib system
            base_face_path = Path(FACE_ASSETS_DIR) / "realistic_face.jpg"
            
            if self.dlib_face_animator and base_face_path.exists():
                print(f"🎭 DLIB: Loading base face from {base_face_path}")
                if self.dlib_face_animator.load_base_face(str(base_face_path)):
                    print("✅ DLIB: Base face loaded with facial landmarks detected")
                    self.current_face = self.dlib_face_animator.base_face.copy()
                    return
                else:
                    print("❌ DLIB: Failed to load base face, falling back to audio-driven")
                    self.dlib_face_animator = None
            
            # Fallback to audio-driven face system
            if self.audio_driven_face and base_face_path.exists():
                print(f"🎭 AUDIO-DRIVEN: Loading base face from {base_face_path}")
                if self.audio_driven_face.load_base_face(str(base_face_path)):
                    print("✅ AUDIO-DRIVEN: Base face loaded")
                    self.current_face = self.audio_driven_face.base_face.copy()
                    return
            
            # Final fallback - load traditional face images
            print("⚠️ FALLBACK: Loading traditional face images")
            self.face_images = {}
            for face_type in ['base', 'mouth_closed', 'mouth_open', 'mouth_wide']:
                face_path = Path(FACE_ASSETS_DIR) / f"{face_type}.jpg"
                if face_path.exists():
                    face_image = cv2.imread(str(face_path))
                    if face_image is not None:
                        self.face_images[face_type] = face_image
                        print(f"✅ TRADITIONAL: Loaded {face_type}")
            
            # Set current face from traditional images
            if hasattr(self, 'face_images') and self.face_images:
                self.current_face = self.face_images.get('mouth_closed', self.face_images.get('base'))
                print(f"✅ TRADITIONAL: Using {len(self.face_images)} face images")
            else:
                print("❌ ERROR: No face assets could be loaded!")
                
        except Exception as e:
            self.logger.error(f"Error loading face assets: {e}")
            print(f"❌ ERROR: Failed to load face assets: {e}")
    
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
        """Animate speaking using best available system: Wav2Lip AI > dlib > audio-driven > phonemes"""
        try:
            self.logger.info("animate_speaking_with_audio called")
            
            # Try Wav2Lip AI first (highest quality)
            if self.wav2lip_animator:
                self.logger.info(f"Using Wav2Lip AI with {len(audio_data)} bytes of audio")
                await self._animate_with_wav2lip(audio_data, phonemes)
                return
            
            # Fallback to dlib system
            elif self.dlib_face_animator:
                self.logger.info(f"Using dlib system with {len(audio_data)} bytes of audio")
                await self._animate_with_dlib(audio_data, phonemes)
                return
                
            # Fallback to audio-driven system
            elif self.audio_driven_face:
                self.logger.info(f"Using audio-driven face manipulation with {len(audio_data)} bytes of audio")
                await self._animate_with_audio_driven(audio_data, phonemes)
                return
            
            # Final fallback to phoneme animation
            else:
                self.logger.warning("No advanced animation systems available, falling back to phoneme animation")
                await self.animate_speaking(phonemes)
                return
        except Exception as e:
            self.logger.error(f"Speaking animation error: {e}")
            self.is_speaking = False
    
    async def _animate_with_wav2lip(self, audio_data: bytes, phonemes: List[dict]):
        """Animate using Wav2Lip AI"""
        try:
            # Initialize Wav2Lip models if needed
            if not await self.wav2lip_animator.initialize():
                self.logger.error("Failed to initialize Wav2Lip models")
                await self._animate_with_dlib(audio_data, phonemes)
                return
            
            # Load base face if not already loaded
            base_face_path = Path(FACE_ASSETS_DIR) / "mouth_closed.png"
            if not self.wav2lip_animator.load_base_face(str(base_face_path)):
                self.logger.error("Failed to load base face for Wav2Lip")
                await self._animate_with_dlib(audio_data, phonemes)
                return
            
            self.is_speaking = True
            self.current_state = "speaking"
            
            # Calculate frame timing
            total_duration = sum(p.get('duration', 0) for p in phonemes) / 1000.0
            frame_rate = 25  # Wav2Lip works well at 25 FPS
            frame_duration = 1.0 / frame_rate
            total_frames = int(total_duration * frame_rate)
            
            self.logger.info(f"Wav2Lip animation: {total_duration:.2f}s, {frame_rate} FPS, {total_frames} frames")
            
            # Convert audio to numpy array
            import numpy as np
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            samples_per_frame = len(audio_array) // total_frames if total_frames > 0 else len(audio_array)
            
            # Animation loop
            for frame in range(total_frames):
                start_idx = frame * samples_per_frame
                end_idx = min(start_idx + samples_per_frame, len(audio_array))
                audio_chunk = audio_array[start_idx:end_idx]
                
                # Generate lip-synced frame
                lip_sync_frame = await self.wav2lip_animator.generate_lip_sync_frame(audio_chunk)
                
                if lip_sync_frame is not None:
                    # Clear screen and display
                    self.display_manager.clear_screen()
                    self.display_manager.display_image(lip_sync_frame, (0, 0))
                    self.display_manager.update_display()
                    
                    print(f"✅ WAV2LIP: Frame {frame} displayed successfully")
                
                # Control frame rate
                await asyncio.sleep(frame_duration)
                
        except Exception as e:
            self.logger.error(f"Wav2Lip animation error: {e}")
            # Fallback to dlib
            await self._animate_with_dlib(audio_data, phonemes)
        finally:
            self.is_speaking = False
    
    async def _animate_with_dlib(self, audio_data: bytes, phonemes: List[dict]):
        """Animate using dlib facial landmarks"""
        try:
            if not self.dlib_face_animator:
                await self._animate_with_audio_driven(audio_data, phonemes)
                return
                
            self.is_speaking = True
            self.current_state = "speaking"
            
            # Calculate total duration and frame rate
            total_duration = sum(p.get('duration', 0) for p in phonemes) / 1000.0
            frame_rate = 15  # 15 FPS for smooth animation
            frame_duration = 1.0 / frame_rate
            total_frames = int(total_duration * frame_rate)
            
            self.logger.info(f"dlib animation: {total_duration:.2f}s, {frame_rate} FPS, {total_frames} frames")
            
            # Convert full audio once
            audio_array = self.dlib_face_animator._bytes_to_audio_array(audio_data)
            if len(audio_array) == 0:
                self.logger.error("Failed to convert audio, falling back to phoneme animation")
                await self.animate_speaking(phonemes)
                return
            
            # Calculate samples per frame with minimum size for analysis
            raw_samples_per_frame = len(audio_array) // total_frames if total_frames > 0 else len(audio_array)
            samples_per_frame = max(1024, raw_samples_per_frame)  # Ensure minimum 1024 samples for better frequency analysis
            
            print(f"🎭 CHUNK SIZE FIX: raw_samples_per_frame={raw_samples_per_frame}, fixed_samples_per_frame={samples_per_frame}")
            print(f"🎭 AUDIO ANALYSIS: total_audio_samples={len(audio_array)}, total_frames={total_frames}")
            print(f"🎭 CHUNKING MATH: samples_per_frame={samples_per_frame}, total_samples_needed={samples_per_frame * total_frames}")
            print(f"🎭 CHUNKING CHECK: Will run out of audio at frame {len(audio_array) // samples_per_frame}")
            self.logger.info(f"Processing {samples_per_frame} samples per frame (minimum 1024 for analysis)")
            
            # Real-time animation loop using dlib
            for frame in range(total_frames):
                # Remove the incorrect check - frame should never exceed total_frames
                # The original check was comparing frame number to audio sample count!
                
                # Extract audio chunk for this frame
                start_idx = frame * samples_per_frame
                end_idx = min(start_idx + samples_per_frame, len(audio_array))
                audio_chunk = audio_array[start_idx:end_idx]
                
                # CRITICAL FIX: Don't break if audio chunk is empty, just use the last available audio
                if len(audio_chunk) == 0:
                    print(f"⚠️ AUDIO CHUNK EMPTY: Frame {frame}, using last available audio")
                    # Use the last available audio chunk instead of breaking
                    if frame > 0:
                        start_idx = max(0, len(audio_array) - samples_per_frame)
                        audio_chunk = audio_array[start_idx:]
                    else:
                        # If first frame is empty, use a small chunk from the beginning
                        audio_chunk = audio_array[:min(1024, len(audio_array))]
                
                # STOP ANIMATION WHEN AUDIO ENDS - Don't keep moving after speech is done
                # Check if we've gone past the actual audio duration
                audio_duration_seconds = len(audio_array) / 22050  # 22050 Hz sample rate
                current_time_seconds = frame / 15.0  # 15 FPS
                
                if current_time_seconds > audio_duration_seconds + 0.5:  # Stop 0.5 seconds after audio ends
                    print(f"⏹️ AUDIO ENDED: Stopping animation at frame {frame} (audio ended at {audio_duration_seconds:.2f}s, current time {current_time_seconds:.2f}s)")
                    break
            
                # Generate face using dlib system
                face = self.dlib_face_animator.generate_face_for_audio_chunk(audio_chunk)
            
                # Debug output every 5 frames
                if frame % 5 == 0:
                    print(f"🎭 ANIMATION DEBUG: Frame {frame}, audio chunk: {len(audio_chunk)} samples, face shape: {face.shape}")
            
                # 🔍 DISPLAY DEBUG: Show the face on screen using existing logic
                try:
                    print(f"🖥️ DISPLAY DEBUG: Frame {frame} - About to display face...")
                    
                    # 🔧 MOUTH-FOCUSED DISPLAY: Crop around mouth then scale to ensure visibility
                    # Original face: (1536, 1024, 3), mouth at (512, 1152)
                    import cv2
                    
                    # Crop around the face area - use image center instead of invalid mouth position
                    face_center_x = face.shape[1] // 2  # Center of image width
                    face_center_y = face.shape[0] // 2  # Center of image height
                    
                    # Define crop area large enough to show the full face
                    crop_width = min(1400, face.shape[1])   # Nearly full width
                    crop_height = min(1000, face.shape[0])  # Nearly full height
                    
                    # Calculate crop bounds
                    x1 = max(0, face_center_x - crop_width // 2)
                    x2 = min(face.shape[1], x1 + crop_width)
                    y1 = max(0, face_center_y - crop_height // 2)  # Center on face
                    y2 = min(face.shape[0], y1 + crop_height)
                    
                    # Ensure we got the right dimensions
                    if x2 - x1 < crop_width:
                        x1 = max(0, x2 - crop_width)
                    if y2 - y1 < crop_height:
                        y1 = max(0, y2 - crop_height)
                    
                    # Crop the face
                    cropped_face = face[y1:y2, x1:x2]
                    
                    # Scale to display size
                    target_height = 600
                    aspect_ratio = cropped_face.shape[1] / cropped_face.shape[0]
                    target_width = int(target_height * aspect_ratio)
                    
                    scaled_face = cv2.resize(cropped_face, (target_width, target_height))
                    
                    # Debug crop info
                    print(f"🔧 CROP DEBUG: Original {face.shape} -> Cropped {cropped_face.shape} -> Scaled {scaled_face.shape}")
                    
                    # Calculate center positions for debugging
                    face_center = (face.shape[1]//2, face.shape[0]//2)
                    crop_center = (cropped_face.shape[1]//2, cropped_face.shape[0]//2)
                    scaled_center = (scaled_face.shape[1]//2, scaled_face.shape[0]//2)
                    
                    print(f"📍 CENTER DEBUG: Face center {face_center} -> Crop {crop_center} -> Scaled {scaled_center}")
                    print(f"👁️ VIEW DEBUG: Full face view centered at {scaled_center}")
                    
                    # Calculate screen position to center the face
                    screen_width, screen_height = self.display_manager.get_screen_size()
                    screen_center_x = (screen_width - target_width) // 2
                    screen_center_y = (screen_height - target_height) // 2
                    screen_position = (screen_center_x, screen_center_y)
                    
                    # Clear screen and display in one operation to avoid double flips
                    self.display_manager.clear_screen()
                    self.display_manager.display_image(scaled_face, screen_position)
                    
                    print(f"✅ DISPLAY DEBUG: Frame {frame} - Full face displayed and screen updated successfully")
                    
                except Exception as e:
                    print(f"❌ DISPLAY ERROR: Frame {frame} - {e}")
                    # Fallback to simple display
                    self.display_manager.clear_screen()
                    self.display_manager.display_image(face, (0, 0))
                
                await asyncio.sleep(frame_duration)
                
        except Exception as e:
            self.logger.error(f"dlib animation error: {e}")
            await self._animate_with_audio_driven(audio_data, phonemes)
        finally:
            self.is_speaking = False
    
    async def _animate_with_audio_driven(self, audio_data: bytes, phonemes: List[dict]):
        """Animate using original audio-driven system"""
        try:
            if not self.audio_driven_face:
                await self.animate_speaking(phonemes)
                return
                
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
            
            # Calculate samples per frame with minimum size for analysis
            raw_samples_per_frame = len(audio_array) // total_frames if total_frames > 0 else len(audio_array)
            samples_per_frame = max(1024, raw_samples_per_frame)  # Ensure minimum 1024 samples for better frequency analysis
            
            print(f"🎭 CHUNK SIZE FIX: raw_samples_per_frame={raw_samples_per_frame}, fixed_samples_per_frame={samples_per_frame}")
            self.logger.info(f"Processing {samples_per_frame} samples per frame (minimum 1024 for analysis)")
            
            # Real-time animation loop
            for frame in range(total_frames):
                # Remove the incorrect check - frame should never exceed total_frames
                # The original check was comparing frame number to audio sample count!
                
                # Extract audio chunk for this frame
                start_idx = frame * samples_per_frame
                end_idx = min(start_idx + samples_per_frame, len(audio_array))
                audio_chunk = audio_array[start_idx:end_idx]
                
                # CRITICAL FIX: Don't break if audio chunk is empty, just use the last available audio
                if len(audio_chunk) == 0:
                    print(f"⚠️ AUDIO CHUNK EMPTY: Frame {frame}, using last available audio")
                    # Use the last available audio chunk instead of breaking
                    if frame > 0:
                        start_idx = max(0, len(audio_array) - samples_per_frame)
                        audio_chunk = audio_array[start_idx:]
                    else:
                        # If first frame is empty, use a small chunk from the beginning
                        audio_chunk = audio_array[:min(1024, len(audio_array))]
            
                # Generate face for this chunk
                face = self._generate_face_for_chunk(audio_chunk)
            
                # Debug output every 5 frames
                if frame % 5 == 0:
                    print(f"🎭 ANIMATION DEBUG: Frame {frame}, audio chunk: {len(audio_chunk)} samples, face shape: {face.shape}")
            
                # 🔍 DISPLAY DEBUG: Show the face on screen
                try:
                    print(f"🖥️ DISPLAY DEBUG: Frame {frame} - About to display face...")
                    
                    # 🔧 MOUTH-FOCUSED DISPLAY: Crop around mouth then scale to ensure visibility
                    # Original face: (1536, 1024, 3), mouth at (512, 1152)
                    import cv2
                    
                    # Crop around the face area - use image center instead of invalid mouth position
                    face_center_x = face.shape[1] // 2  # Center of image width
                    face_center_y = face.shape[0] // 2  # Center of image height
                    
                    # Define crop area large enough to show the full face
                    crop_width = min(1400, face.shape[1])   # Nearly full width
                    crop_height = min(1000, face.shape[0])  # Nearly full height
                    
                    # Calculate crop bounds
                    x1 = max(0, face_center_x - crop_width // 2)
                    x2 = min(face.shape[1], x1 + crop_width)
                    y1 = max(0, face_center_y - crop_height // 2)  # Center on face
                    y2 = min(face.shape[0], y1 + crop_height)
                    
                    # Ensure we got the right dimensions
                    if x2 - x1 < crop_width:
                        x1 = max(0, x2 - crop_width)
                    if y2 - y1 < crop_height:
                        y1 = max(0, y2 - crop_height)
                    
                    # Crop the face to show full face area
                    face_crop = face[y1:y2, x1:x2]
                    
                    # Scale the full-face crop to fit screen nicely
                    target_height = 600  # Good viewing size for full face
                    scale_factor = target_height / face_crop.shape[0]
                    new_width = int(face_crop.shape[1] * scale_factor)
                    new_height = int(face_crop.shape[0] * scale_factor)
                    
                    scaled_face = cv2.resize(face_crop, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                    
                    # Calculate face center in the cropped view (no mouth overlay needed)
                    face_center_in_crop_x = face_center_x - x1
                    face_center_in_crop_y = face_center_y - y1
                    scaled_center_x = int(face_center_in_crop_x * scale_factor)
                    scaled_center_y = int(face_center_in_crop_y * scale_factor)
                    
                    print(f"🔧 CROP DEBUG: Original {face.shape} -> Cropped {face_crop.shape} -> Scaled {scaled_face.shape}")
                    print(f"🎯 CENTER DEBUG: Face center ({face_center_x},{face_center_y}) -> Crop ({face_center_in_crop_x},{face_center_in_crop_y}) -> Scaled ({scaled_center_x},{scaled_center_y})")
                    print(f"👁️ VIEW DEBUG: Full face view centered at ({scaled_center_x},{scaled_center_y})")
                    
                    # Clear screen and display the full face view centered
                    self.display_manager.clear_screen()
                    
                    screen_width, screen_height = self.display_manager.get_screen_size()
                    screen_center_x = (screen_width - new_width) // 2
                    screen_center_y = (screen_height - new_height) // 2
                    screen_position = (screen_center_x, screen_center_y)
                    self.display_manager.display_image(scaled_face, screen_position)
                    
                    print(f"✅ DISPLAY DEBUG: Frame {frame} - Full face displayed and screen updated successfully")
                except Exception as e:
                    print(f"❌ DISPLAY DEBUG: Frame {frame} - Display failed: {e}")
                    import traceback
                    print(f"❌ DISPLAY TRACEBACK: {traceback.format_exc()}")
            
                # Control frame rate (15 FPS)
                await asyncio.sleep(1/15)
            
            # Return to idle
            self.is_speaking = False
            self.current_state = "idle"
            
        except Exception as e:
            self.logger.error(f"Error in audio-driven speaking animation: {e}")
            self.logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
            # Fallback to regular animation
            await self.animate_speaking(phonemes)
    
    def _generate_face_for_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Generate face for a single audio chunk using best available system"""
        try:
            print(f"🎬 CHUNK DEBUG: _generate_face_for_chunk called with {len(audio_chunk)} samples")
            
            # Use the actual TTS audio chunk for animation
            # This will make the mouth respond to the AI-generated speech
            print(f"🎵 TTS AUDIO: Using AI-generated speech for animation")
            
            # Convert audio chunk to bytes for compatibility
            audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
            duration = len(audio_chunk) / 22050
            
            # Try dlib system first (most accurate)
            if self.dlib_face_animator:
                print(f"✅ DLIB: Using facial landmark animation")
                face = self.dlib_face_animator.generate_face_for_audio_chunk(audio_chunk)
                print(f"✅ DLIB: Generated face with shape: {face.shape}")
                return face
            
            # Fallback to audio-driven system  
            elif self.audio_driven_face and hasattr(self.audio_driven_face, 'generate_face_from_audio'):
                print(f"✅ AUDIO-DRIVEN: Using audio-driven animation")
                face = self.audio_driven_face.generate_face_from_audio(audio_bytes, duration)
                print(f"✅ AUDIO-DRIVEN: Generated face with shape: {face.shape}")
                return face
            
            # Final fallback to static face
            else:
                print(f"⚠️ STATIC: Using static face (no animation systems available)")
                return self.current_face.copy() if self.current_face is not None else np.zeros((512, 512, 3), dtype=np.uint8)
                
        except Exception as e:
            print(f"❌ CHUNK DEBUG: Error in _generate_face_for_chunk: {e}")
            import traceback
            print(f"❌ CHUNK TRACEBACK: {traceback.format_exc()}")
            self.logger.error(f"Error generating face for chunk: {e}")
            return self.current_face.copy() if self.current_face is not None else np.zeros((512, 512, 3), dtype=np.uint8)
    
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