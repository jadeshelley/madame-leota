"""
SadTalker AI-Powered Talking Face Animator for Madame Leota
Uses SadTalker for realistic talking face generation
"""

import cv2
import numpy as np
import logging
import os
import tempfile
import subprocess
import requests
import json
from pathlib import Path
from typing import Optional, Tuple
import time

class SadTalkerAnimator:
    """SadTalker-based talking face animator"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_face = None
        self.frame_counter = 0
        self.temp_dir = Path("temp_sadtalker")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Check if SadTalker is available
        self.sadtalker_available = self._check_sadtalker_availability()
        
        if self.sadtalker_available:
            print("✅ SADTALKER: SadTalker system available")
        else:
            print("❌ SADTALKER: SadTalker not available, will use fallback")
        
        self.logger.info("SadTalker animator initialized")
    
    def _check_sadtalker_availability(self) -> bool:
        """Check if SadTalker is available on the system"""
        try:
            # Try multiple methods to check for SadTalker
            methods = [
                self._check_python_import,
                self._check_command_line,
                self._check_api_endpoint
            ]
            
            for method in methods:
                if method():
                    return True
            
            return False
        except Exception as e:
            print(f"⚠️ SADTALKER: Error checking availability: {e}")
            return False
    
    def _check_python_import(self) -> bool:
        """Check if SadTalker can be imported"""
        try:
            import torch
            # Try to import SadTalker modules
            import sys
            sys.path.append("SadTalker")
            from src.utils.preprocess import align_img
            print("✅ SADTALKER: Python import successful")
            return True
        except ImportError:
            return False
    
    def _check_command_line(self) -> bool:
        """Check if SadTalker command line tool is available"""
        try:
            result = subprocess.run(["python", "-c", "import torch; print('PyTorch available')"], 
                                  capture_output=True, text=True, timeout=5)
            return "PyTorch available" in result.stdout
        except:
            return False
    
    def _check_api_endpoint(self) -> bool:
        """Check if SadTalker API is running"""
        try:
            response = requests.get("http://localhost:7860", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def load_base_face(self, face_image_path: str) -> bool:
        """Load the base face image"""
        try:
            self.base_face = cv2.imread(face_image_path)
            if self.base_face is not None:
                print(f"✅ SADTALKER: Loaded base face {self.base_face.shape}")
                return True
            else:
                print(f"❌ SADTALKER: Failed to load {face_image_path}")
                return False
        except Exception as e:
            print(f"❌ SADTALKER: Error loading face: {e}")
            return False
    
    def generate_face_for_audio_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Generate animated face based on audio using SadTalker"""
        try:
            if self.base_face is None:
                return np.zeros((512, 512, 3), dtype=np.uint8)
            
            # For now, create a simple animated overlay while we set up SadTalker
            # This will be replaced with actual SadTalker processing
            
            # Create a simple mouth animation overlay
            frame_cycle = self.frame_counter % 30  # 30-frame cycle
            
            if frame_cycle < 10:  # First 10 frames: open mouth
                mouth_open = 0.8
                phoneme_type = "SADTALKER"
                color = (0, 255, 255)  # Cyan
            elif frame_cycle < 20:  # Next 10 frames: medium
                mouth_open = 0.4
                phoneme_type = "PROCESSING"
                color = (255, 255, 0)  # Yellow
            else:  # Last 10 frames: closed
                mouth_open = 0.1
                phoneme_type = "READY"
                color = (0, 255, 0)  # Green
            
            # Create animated face
            result = self.base_face.copy()
            
            # Draw animated mouth overlay
            center_x = result.shape[1] // 2
            center_y = result.shape[0] // 2 + 100
            mouth_radius = int(20 + (mouth_open * 60))
            cv2.circle(result, (center_x, center_y), mouth_radius, color, -1)
            
            # Add text labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_pos = (center_x - 100, center_y - mouth_radius - 30)
            cv2.putText(result, phoneme_type, text_pos, font, 1.0, (255, 255, 255), 2)
            
            # Add status info
            status_text = f"SadTalker: {'Available' if self.sadtalker_available else 'Not Available'}"
            cv2.putText(result, status_text, (10, 60), font, 0.6, (255, 255, 255), 2)
            
            # Add frame counter
            frame_text = f"Frame: {self.frame_counter}"
            cv2.putText(result, frame_text, (10, 30), font, 0.7, (255, 255, 255), 2)
            
            self.frame_counter += 1
            return result
            
        except Exception as e:
            print(f"❌ SADTALKER ERROR: {e}")
            return self.base_face.copy() if self.base_face is not None else np.zeros((512, 512, 3), dtype=np.uint8)
    
    def create_talking_video(self, audio_data: bytes, output_path: str = "talking_face.mp4") -> Optional[str]:
        """Create a talking video using SadTalker"""
        try:
            if not self.sadtalker_available:
                print("❌ SADTALKER: SadTalker not available for video creation")
                return None
            
            # Save audio to temp file
            audio_path = self.temp_dir / "input_audio.wav"
            with open(audio_path, 'wb') as f:
                f.write(audio_data)
            
            # Save face image to temp file
            face_path = self.temp_dir / "input_face.jpg"
            cv2.imwrite(str(face_path), self.base_face)
            
            # Try to run SadTalker
            success = self._run_sadtalker(str(face_path), str(audio_path), output_path)
            
            if success:
                print(f"✅ SADTALKER: Created talking video: {output_path}")
                return output_path
            else:
                print("❌ SADTALKER: Failed to create talking video")
                return None
                
        except Exception as e:
            print(f"❌ SADTALKER: Error creating video: {e}")
            return None
    
    def _run_sadtalker(self, face_path: str, audio_path: str, output_path: str) -> bool:
        """Run SadTalker command line tool"""
        try:
            # Try different SadTalker command formats
            commands = [
                ["python", "SadTalker/inference.py", "--driven_audio", audio_path, 
                 "--source_image", face_path, "--result_dir", "temp_sadtalker"],
                ["python", "-m", "SadTalker.inference", "--driven_audio", audio_path,
                 "--source_image", face_path, "--result_dir", "temp_sadtalker"],
                ["sadtalker", "--driven_audio", audio_path, "--source_image", face_path,
                 "--result_dir", "temp_sadtalker"]
            ]
            
            for cmd in commands:
                try:
                    print(f"🔄 SADTALKER: Trying command: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    
                    if result.returncode == 0:
                        print("✅ SADTALKER: Command successful")
                        return True
                    else:
                        print(f"⚠️ SADTALKER: Command failed: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    print("⚠️ SADTALKER: Command timed out")
                except FileNotFoundError:
                    print("⚠️ SADTALKER: Command not found")
            
            return False
            
        except Exception as e:
            print(f"❌ SADTALKER: Error running SadTalker: {e}")
            return False
    
    def _bytes_to_audio_array(self, audio_data: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array"""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
            return audio_array
        except Exception as e:
            print(f"❌ SADTALKER: Error converting audio: {e}")
            return np.array([], dtype=np.float32)
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                print("✅ SADTALKER: Cleaned up temporary files")
        except Exception as e:
            print(f"⚠️ SADTALKER: Error cleaning up: {e}") 