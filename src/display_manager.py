"""
Display Manager for Madame Leota
Handles projector output and display management
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional
import pygame
from config import *

class DisplayManager:
    def __init__(self, width: int = None, height: int = None, fullscreen: bool = None):
        self.logger = logging.getLogger(__name__)
        
        # Use config values as defaults
        if width is None:
            width = PROJECTOR_WIDTH
        if height is None:
            height = PROJECTOR_HEIGHT  
        if fullscreen is None:
            fullscreen = FULLSCREEN
        
        # Display settings
        self.screen_width = width
        self.screen_height = height
        self.fullscreen = fullscreen
        self.background_color = BACKGROUND_COLOR
        
        # Face display dimensions for scaling
        self.face_display_width = int(width * 0.6)  # 60% of screen width
        self.face_display_height = int(height * 0.8)  # 80% of screen height
        
        # Center coordinates
        self.center_x = width // 2
        self.center_y = height // 2
        
        # Initialize pygame display
        pygame.init()
        
        # Set up display
        if fullscreen:
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height), 
                pygame.FULLSCREEN
            )
        else:
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
        
        pygame.display.set_caption("Madame Leota")
        
        # Display properties
        self.background_color = (0, 0, 0)  # Black background
        self.center_x = self.screen_width // 2
        self.center_y = self.screen_height // 2
        
        # Frame rate control
        self.clock = pygame.time.Clock()
        
        self.logger.info(f"Display initialized: {self.screen_width}x{self.screen_height}")
    
    def clear_screen(self):
        """Clear the screen with background color"""
        self.screen.fill(self.background_color)
    
    def display_image(self, image: np.ndarray, position: Tuple[int, int]):
        """Display a numpy image on the screen"""
        try:
            if image is None or image.size == 0:
                self.logger.warning("Invalid image provided to display_image")
                return
            
            # Ensure image is in correct format for pygame
            if len(image.shape) == 3:
                # Convert BGR to RGB if needed and ensure uint8
                if image.shape[2] == 3:
                    image_rgb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image.astype(np.uint8)
            else:
                self.logger.error(f"Unsupported image shape: {image.shape}")
                return
            
            # Ensure the image dimensions are valid
            if image_rgb.shape[0] < 1 or image_rgb.shape[1] < 1:
                self.logger.warning(f"Invalid image dimensions: {image_rgb.shape}")
                return
            
            # Ensure image is contiguous in memory for pygame
            if not image_rgb.flags['C_CONTIGUOUS']:
                image_rgb = np.ascontiguousarray(image_rgb)
            
            # Create pygame surface with error handling
            try:
                surface = pygame.surfarray.make_surface(image_rgb.swapaxes(0, 1))
            except (ValueError, TypeError) as surf_error:
                self.logger.error(f"Surface creation failed: {surf_error}, image shape: {image_rgb.shape}, dtype: {image_rgb.dtype}")
                # Try alternative method
                try:
                    # Convert to PIL Image first, then to pygame
                    from PIL import Image as PILImage
                    pil_image = PILImage.fromarray(image_rgb)
                    mode = pil_image.mode
                    size = pil_image.size
                    raw_data = pil_image.tobytes()
                    surface = pygame.image.fromstring(raw_data, size, mode)
                except Exception as pil_error:
                    self.logger.error(f"PIL fallback failed: {pil_error}")
                    return
            
            # Blit to screen
            self.screen.blit(surface, position)
            
        except Exception as e:
            self.logger.error(f"Error displaying image: {e}")
            self.logger.error(f"Image info: shape={image.shape if image is not None else 'None'}, dtype={image.dtype if image is not None else 'None'}")
    
    def display_face(self, face_image: np.ndarray):
        """Display Madame Leota's face, scaled and positioned for head form"""
        try:
            # Scale the face image
            scaled_face = self._scale_face_for_projection(face_image)
            
            # Position for optimal projection onto head form
            face_position = self._calculate_face_position(scaled_face)
            
            # Display the face
            self.display_image(scaled_face, face_position)
            
        except Exception as e:
            self.logger.error(f"Face display error: {e}")
    
    def _scale_face_for_projection(self, face_image: np.ndarray) -> np.ndarray:
        """Scale face image for optimal projection display"""
        try:
            if face_image is None or face_image.size == 0:
                self.logger.error("Invalid face image provided for scaling")
                return face_image
            
            # Validate image shape and data
            if len(face_image.shape) != 3 or face_image.shape[2] != 3:
                self.logger.error(f"Invalid face image shape: {face_image.shape}")
                return face_image
            
            h, w = face_image.shape[:2]
            if h < 1 or w < 1:
                self.logger.error(f"Invalid face image dimensions: {h}x{w}")
                return face_image
            
            # Ensure image data is valid
            if not np.isfinite(face_image).all():
                self.logger.warning("Face image contains invalid values, cleaning...")
                face_image = np.nan_to_num(face_image, nan=0, posinf=255, neginf=0)
            
            # Ensure proper data type
            if face_image.dtype != np.uint8:
                face_image = np.clip(face_image, 0, 255).astype(np.uint8)
            
            # Calculate target size while maintaining aspect ratio
            current_ratio = w / h
            target_ratio = self.face_display_width / self.face_display_height
            
            if current_ratio > target_ratio:
                # Image is wider - fit to width
                new_width = self.face_display_width
                new_height = int(self.face_display_width / current_ratio)
            else:
                # Image is taller - fit to height  
                new_height = self.face_display_height
                new_width = int(self.face_display_height * current_ratio)
            
            # Ensure minimum dimensions
            new_width = max(1, new_width)
            new_height = max(1, new_height)
            
            # Perform resize with error handling
            try:
                scaled_face = cv2.resize(
                    face_image, 
                    (new_width, new_height), 
                    interpolation=cv2.INTER_LINEAR
                )
            except cv2.error as cv_error:
                self.logger.error(f"OpenCV resize failed: {cv_error}")
                # Try with different interpolation
                try:
                    scaled_face = cv2.resize(
                        face_image, 
                        (new_width, new_height), 
                        interpolation=cv2.INTER_NEAREST
                    )
                except Exception as fallback_error:
                    self.logger.error(f"Fallback resize failed: {fallback_error}")
                    return face_image
            
            return scaled_face
            
        except Exception as e:
            self.logger.error(f"Face scaling error: {e}")
            self.logger.error(f"Face info: shape={face_image.shape if face_image is not None else 'None'}, dtype={face_image.dtype if face_image is not None else 'None'}")
            return face_image
    
    def _calculate_face_position(self, face_image: np.ndarray) -> Tuple[int, int]:
        """Calculate optimal position for face projection"""
        try:
            height, width = face_image.shape[:2]
            
            # Center horizontally, position slightly higher vertically for head form
            x = (self.screen_width - width) // 2
            y = (self.screen_height - height) // 2 - int(height * 0.1)  # Slightly higher
            
            return (x, y)
            
        except Exception as e:
            self.logger.error(f"Position calculation error: {e}")
            return (0, 0)
    
    def add_glow_effect(self, image: np.ndarray, intensity: float = 0.3) -> np.ndarray:
        """Add a mystical glow effect to the image"""
        try:
            # Create a blurred version for glow
            blurred = cv2.GaussianBlur(image, (21, 21), 0)
            
            # Blend original with blurred for glow effect
            glowing = cv2.addWeighted(image, 1.0, blurred, intensity, 0)
            
            return glowing
            
        except Exception as e:
            self.logger.error(f"Glow effect error: {e}")
            return image
    
    def add_fade_effect(self, image: np.ndarray, fade_factor: float) -> np.ndarray:
        """Add fade in/out effect"""
        try:
            fade_factor = max(0.0, min(1.0, fade_factor))
            faded = cv2.convertScaleAbs(image, alpha=fade_factor, beta=0)
            return faded
            
        except Exception as e:
            self.logger.error(f"Fade effect error: {e}")
            return image
    
    def update_display(self, face_image=None):
        """Update the display with current content"""
        try:
            if face_image is not None:
                self.display_face(face_image)
            
            # Check if pygame display is properly initialized
            if not pygame.get_init() or not pygame.display.get_init():
                self.logger.warning("Pygame display not properly initialized, skipping update")
                return
            
            # Check if we have a valid surface
            screen = pygame.display.get_surface()
            if screen is None:
                self.logger.warning("No valid pygame surface available, skipping update")
                return
            
            # Safe display update with error handling
            pygame.display.flip()
            
        except pygame.error as e:
            self.logger.error(f"Pygame display error: {e}")
        except Exception as e:
            self.logger.error(f"Display update error: {e}")
            # Don't re-raise the exception, just log it
    
    def handle_events(self) -> bool:
        """Handle pygame events, returns False if quit requested"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_f:
                    # Toggle fullscreen
                    pygame.display.toggle_fullscreen()
        
        return True
    
    def get_screen_size(self) -> Tuple[int, int]:
        """Get current screen dimensions"""
        return (self.screen_width, self.screen_height)
    
    def capture_screen(self) -> np.ndarray:
        """Capture current screen content as numpy array"""
        try:
            # Get surface as array
            screen_array = pygame.surfarray.array3d(self.screen)
            
            # Convert from pygame format to OpenCV format
            screen_image = np.transpose(screen_array, (1, 0, 2))
            screen_image = cv2.cvtColor(screen_image, cv2.COLOR_RGB2BGR)
            
            return screen_image
            
        except Exception as e:
            self.logger.error(f"Screen capture error: {e}")
            return np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
    
    def cleanup(self):
        """Cleanup display resources"""
        try:
            pygame.quit()
            self.logger.info("Display Manager cleaned up")
            
        except Exception as e:
            self.logger.error(f"Display cleanup error: {e}") 