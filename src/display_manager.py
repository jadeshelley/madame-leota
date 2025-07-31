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
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize pygame display
        pygame.init()
        
        # Set up display
        self.screen_width = PROJECTOR_WIDTH
        self.screen_height = PROJECTOR_HEIGHT
        
        if FULLSCREEN:
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
    
    def display_image(self, image: np.ndarray, position: Optional[Tuple[int, int]] = None):
        """Display an image on screen"""
        try:
            # Convert OpenCV image (BGR) to pygame surface (RGB)
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Convert to pygame surface
            surface = pygame.surfarray.make_surface(image_rgb.swapaxes(0, 1))
            
            # Position the image
            if position is None:
                # Center the image
                image_rect = surface.get_rect()
                image_rect.center = (self.center_x, self.center_y)
                position = image_rect.topleft
            
            # Blit to screen
            self.screen.blit(surface, position)
            
        except Exception as e:
            self.logger.error(f"Image display error: {e}")
    
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
        """Scale face image for proper projection onto head form"""
        try:
            height, width = face_image.shape[:2]
            
            # Calculate scale factor based on configuration
            target_width = int(self.screen_width * FACE_SCALE * 0.6)  # 60% of screen width
            scale_factor = target_width / width
            
            target_height = int(height * scale_factor)
            
            # Resize image
            scaled_face = cv2.resize(
                face_image, 
                (target_width, target_height), 
                interpolation=cv2.INTER_LANCZOS4
            )
            
            return scaled_face
            
        except Exception as e:
            self.logger.error(f"Face scaling error: {e}")
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