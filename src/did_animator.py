import requests
import json
import time
import logging
import cv2
import numpy as np
from typing import Optional, Dict, Any
import base64
import os
from pathlib import Path

class DIDAnimator:
    """D-ID API-based face animator for professional-quality lip sync"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key or os.getenv('DID_API_KEY')
        self.base_url = "https://api.d-id.com"
        self.base_face = None
        self.frame_counter = 0
        
        if not self.api_key:
            print("⚠️ DID: No API key provided. Set DID_API_KEY environment variable or pass api_key parameter.")
            print("⚠️ DID: Get free API key at: https://studio.d-id.com/")
        
        self.logger.info("✅ D-ID animator initialized")
    
    def load_base_face(self, face_image_path: str) -> bool:
        """Load the base face image"""
        try:
            self.base_face = cv2.imread(face_image_path)
            if self.base_face is not None:
                print(f"✅ DID: Loaded base face {self.base_face.shape}")
                return True
            else:
                print(f"❌ DID: Failed to load {face_image_path}")
                return False
        except Exception as e:
            print(f"❌ DID: Error loading face: {e}")
            return False
    
    def create_talking_video(self, audio_data: bytes, output_path: str = "talking_face.mp4") -> Optional[str]:
        """Create a talking face video using D-ID API"""
        try:
            if not self.api_key:
                print("❌ DID: No API key available")
                return None
            
            if self.base_face is None:
                print("❌ DID: No base face loaded")
                return None
            
            # Save base face temporarily
            temp_face_path = "temp_face.jpg"
            cv2.imwrite(temp_face_path, self.base_face)
            
            # Save audio temporarily
            temp_audio_path = "temp_audio.wav"
            with open(temp_audio_path, 'wb') as f:
                f.write(audio_data)
            
            # Upload face image
            face_url = self._upload_image(temp_face_path)
            if not face_url:
                return None
            
            # Upload audio
            audio_url = self._upload_audio(temp_audio_path)
            if not audio_url:
                return None
            
            # Create talking video
            video_url = self._create_talk(face_url, audio_url)
            if not video_url:
                return None
            
            # Download video
            success = self._download_video(video_url, output_path)
            
            # Clean up temp files
            self._cleanup_temp_files([temp_face_path, temp_audio_path])
            
            if success:
                print(f"✅ DID: Created talking video: {output_path}")
                return output_path
            else:
                print("❌ DID: Failed to download video")
                return None
                
        except Exception as e:
            print(f"❌ DID ERROR: {e}")
            return None
    
    def _upload_image(self, image_path: str) -> Optional[str]:
        """Upload image to D-ID and get URL"""
        try:
            url = f"{self.base_url}/images"
            headers = {
                "Authorization": f"Basic {self.api_key}",
                "Content-Type": "application/json"
            }
            
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            data = {
                "image": f"data:image/jpeg;base64,{image_data}"
            }
            
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            return result.get('url')
            
        except Exception as e:
            print(f"❌ DID: Error uploading image: {e}")
            return None
    
    def _upload_audio(self, audio_path: str) -> Optional[str]:
        """Upload audio to D-ID and get URL"""
        try:
            url = f"{self.base_url}/audios"
            headers = {
                "Authorization": f"Basic {self.api_key}",
                "Content-Type": "application/json"
            }
            
            with open(audio_path, 'rb') as f:
                audio_data = base64.b64encode(f.read()).decode('utf-8')
            
            data = {
                "audio": f"data:audio/wav;base64,{audio_data}"
            }
            
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            return result.get('url')
            
        except Exception as e:
            print(f"❌ DID: Error uploading audio: {e}")
            return None
    
    def _create_talk(self, image_url: str, audio_url: str) -> Optional[str]:
        """Create talking video using D-ID API"""
        try:
            url = f"{self.base_url}/talks"
            headers = {
                "Authorization": f"Basic {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "image_url": image_url,
                "audio_url": audio_url,
                "config": {
                    "fluent": True,
                    "pad_audio": 0.0
                }
            }
            
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            talk_id = result.get('id')
            
            if not talk_id:
                return None
            
            # Wait for processing to complete
            return self._wait_for_completion(talk_id)
            
        except Exception as e:
            print(f"❌ DID: Error creating talk: {e}")
            return None
    
    def _wait_for_completion(self, talk_id: str, max_wait: int = 60) -> Optional[str]:
        """Wait for video processing to complete"""
        try:
            url = f"{self.base_url}/talks/{talk_id}"
            headers = {
                "Authorization": f"Basic {self.api_key}"
            }
            
            start_time = time.time()
            while time.time() - start_time < max_wait:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                
                result = response.json()
                status = result.get('status')
                
                if status == 'done':
                    return result.get('result_url')
                elif status == 'error':
                    print(f"❌ DID: Processing failed: {result.get('error')}")
                    return None
                
                print(f"⏳ DID: Processing... ({status})")
                time.sleep(2)
            
            print("❌ DID: Processing timeout")
            return None
            
        except Exception as e:
            print(f"❌ DID: Error waiting for completion: {e}")
            return None
    
    def _download_video(self, video_url: str, output_path: str) -> bool:
        """Download video from URL"""
        try:
            response = requests.get(video_url, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True
            
        except Exception as e:
            print(f"❌ DID: Error downloading video: {e}")
            return False
    
    def _cleanup_temp_files(self, file_paths: list):
        """Clean up temporary files"""
        for path in file_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                print(f"⚠️ DID: Error cleaning up {path}: {e}")
    
    def generate_face_for_audio_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Generate face frame (fallback to base face for real-time)"""
        try:
            if self.base_face is None:
                return np.zeros((512, 512, 3), dtype=np.uint8)
            
            # For real-time display, just show base face with overlay
            result = self.base_face.copy()
            
            # Add D-ID info overlay
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(result, "D-ID API", (10, 30), font, 0.7, (0, 255, 255), 2)
            cv2.putText(result, f"Frame: {self.frame_counter}", (10, 60), font, 0.7, (255, 255, 255), 2)
            
            # Add audio info
            if len(audio_chunk) > 0:
                rms = np.sqrt(np.mean(audio_chunk**2))
                cv2.putText(result, f"Audio: {rms:.3f}", (10, 90), font, 0.6, (255, 255, 255), 2)
            
            self.frame_counter += 1
            return result
            
        except Exception as e:
            print(f"❌ DID ERROR: {e}")
            return self.base_face.copy() if self.base_face is not None else np.zeros((512, 512, 3), dtype=np.uint8)
    
    def _bytes_to_audio_array(self, audio_data: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array"""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
            return audio_array
        except Exception as e:
            print(f"❌ DID: Error converting audio: {e}")
            return np.array([], dtype=np.float32) 