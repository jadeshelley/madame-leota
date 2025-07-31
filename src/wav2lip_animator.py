"""
Wav2Lip AI-Powered Lip Sync Animator for Madame Leota
Uses deep learning for realistic lip synchronization
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import tempfile
import librosa
from pathlib import Path
from typing import Optional, Tuple
import urllib.request
from config import *

class Wav2LipAnimator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Wav2Lip using device: {self.device}")
        
        # Model paths
        self.model_dir = Path("models/wav2lip")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Models will be loaded on demand
        self.wav2lip_model = None
        self.face_detect_model = None
        self.base_face = None
        
        # Audio processing parameters
        self.mel_step_size = 16
        self.fps = 25
        self.sample_rate = 16000
        
        self.logger.info("Wav2Lip animator initialized")
    
    async def initialize(self):
        """Initialize models - download if needed"""
        try:
            await self._download_models()
            self._load_models()
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Wav2Lip: {e}")
            return False
    
    async def _download_models(self):
        """Download Wav2Lip models if not present"""
        models_to_download = {
            "wav2lip_gan.pth": "https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/wav2lip_gan.pth",
            "face_detection.pth": "https://github.com/Rudrabha/Wav2Lip/raw/master/face_detection/detection/sfd/s3fd.pth"
        }
        
        for model_name, url in models_to_download.items():
            model_path = self.model_dir / model_name
            if not model_path.exists():
                self.logger.info(f"Downloading {model_name}...")
                try:
                    urllib.request.urlretrieve(url, str(model_path))
                    self.logger.info(f"✅ Downloaded {model_name}")
                except Exception as e:
                    self.logger.error(f"Failed to download {model_name}: {e}")
                    raise
    
    def _load_models(self):
        """Load the Wav2Lip models"""
        try:
            # Load Wav2Lip model
            model_path = self.model_dir / "wav2lip_gan.pth"
            if model_path.exists():
                self.logger.info("Loading Wav2Lip model...")
                checkpoint = torch.load(str(model_path), map_location=self.device)
                self.wav2lip_model = Wav2LipModel()
                self.wav2lip_model.load_state_dict(checkpoint['state_dict'])
                self.wav2lip_model.to(self.device)
                self.wav2lip_model.eval()
                self.logger.info("✅ Wav2Lip model loaded")
            
            # Load face detection model
            face_model_path = self.model_dir / "face_detection.pth"
            if face_model_path.exists():
                self.logger.info("Loading face detection model...")
                self.face_detect_model = self._load_face_detector(str(face_model_path))
                self.logger.info("✅ Face detection model loaded")
                
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            raise
    
    def load_base_face(self, face_image_path: str) -> bool:
        """Load the base face image for animation"""
        try:
            self.base_face = cv2.imread(face_image_path)
            if self.base_face is None:
                self.logger.error(f"Could not load face image: {face_image_path}")
                return False
            
            self.logger.info(f"Base face loaded: {self.base_face.shape}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading base face: {e}")
            return False
    
    async def generate_lip_sync_frame(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """Generate a single lip-synced frame based on audio chunk"""
        try:
            if self.wav2lip_model is None or self.base_face is None:
                self.logger.warning("Models or base face not loaded")
                return self.base_face
            
            # Process audio to mel spectrogram
            mel = self._audio_to_mel(audio_chunk)
            
            # Prepare face input
            face_tensor = self._prepare_face_input(self.base_face)
            
            # Generate lip-synced frame
            with torch.no_grad():
                # Create batch
                face_batch = face_tensor.unsqueeze(0).to(self.device)
                mel_batch = torch.FloatTensor(mel).unsqueeze(0).to(self.device)
                
                # Generate frame
                generated = self.wav2lip_model(mel_batch, face_batch)
                
                # Convert back to numpy
                frame = self._tensor_to_image(generated[0])
                
            return frame
            
        except Exception as e:
            self.logger.error(f"Error generating lip sync frame: {e}")
            return self.base_face
    
    def _audio_to_mel(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Convert audio chunk to mel spectrogram"""
        try:
            # Ensure audio is the right sample rate
            if len(audio_chunk) == 0:
                # Return silent mel spectrogram
                return np.zeros((80, self.mel_step_size), dtype=np.float32)
            
            # Convert to float32 and normalize
            audio = audio_chunk.astype(np.float32)
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            # Generate mel spectrogram
            mel = librosa.feature.melspectrogram(
                y=audio, 
                sr=self.sample_rate,
                n_mels=80,
                fmax=7600,
                hop_length=200,
                win_length=800
            )
            
            # Convert to log scale
            mel = librosa.power_to_db(mel)
            
            # Ensure correct shape
            if mel.shape[1] < self.mel_step_size:
                # Pad if too short
                mel = np.pad(mel, ((0, 0), (0, self.mel_step_size - mel.shape[1])), mode='constant')
            elif mel.shape[1] > self.mel_step_size:
                # Trim if too long
                mel = mel[:, :self.mel_step_size]
            
            return mel.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error processing audio to mel: {e}")
            return np.zeros((80, self.mel_step_size), dtype=np.float32)
    
    def _prepare_face_input(self, face_image: np.ndarray) -> torch.Tensor:
        """Prepare face image for model input"""
        try:
            # Resize to model input size (96x96)
            face = cv2.resize(face_image, (96, 96))
            
            # Convert BGR to RGB
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            
            # Normalize to [-1, 1]
            face = (face / 255.0) * 2.0 - 1.0
            
            # Convert to tensor and add channel dimension
            face_tensor = torch.FloatTensor(face).permute(2, 0, 1)
            
            return face_tensor
            
        except Exception as e:
            self.logger.error(f"Error preparing face input: {e}")
            return torch.zeros((3, 96, 96))
    
    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert model output tensor back to image"""
        try:
            # Move to CPU and convert to numpy
            img = tensor.cpu().detach().numpy()
            
            # Denormalize from [-1, 1] to [0, 255]
            img = (img + 1.0) / 2.0 * 255.0
            
            # Convert from CHW to HWC
            img = np.transpose(img, (1, 2, 0))
            
            # Convert to uint8
            img = np.clip(img, 0, 255).astype(np.uint8)
            
            # Convert RGB back to BGR for OpenCV
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            return img
            
        except Exception as e:
            self.logger.error(f"Error converting tensor to image: {e}")
            return np.zeros((96, 96, 3), dtype=np.uint8)
    
    def _load_face_detector(self, model_path: str):
        """Load face detection model"""
        # Simplified face detector - in real implementation would load S3FD
        # For now, return a placeholder
        self.logger.info("Face detector placeholder loaded")
        return "face_detector_placeholder"
    
    def cleanup(self):
        """Cleanup resources"""
        if self.wav2lip_model:
            del self.wav2lip_model
            self.wav2lip_model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("Wav2Lip animator cleaned up")


class Wav2LipModel(nn.Module):
    """Simplified Wav2Lip model architecture"""
    
    def __init__(self):
        super(Wav2LipModel, self).__init__()
        
        # Audio encoder
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(80, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Face encoder
        self.face_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256 + 128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, mel, face):
        # Encode audio
        audio_feat = self.audio_encoder(mel.permute(0, 2, 1))  # B x 256 x 1
        audio_feat = audio_feat.squeeze(-1).unsqueeze(-1).unsqueeze(-1)  # B x 256 x 1 x 1
        audio_feat = audio_feat.expand(-1, -1, 8, 8)  # B x 256 x 8 x 8
        
        # Encode face
        face_feat = self.face_encoder(face)  # B x 128 x 8 x 8
        
        # Combine features
        combined = torch.cat([audio_feat, face_feat], dim=1)  # B x (256+128) x 8 x 8
        
        # Decode to output
        output = self.decoder(combined)
        
        # Resize to original size
        output = F.interpolate(output, size=(96, 96), mode='bilinear', align_corners=False)
        
        return output


# Global instance
wav2lip_animator = None

def get_wav2lip_animator():
    """Get global Wav2Lip animator instance"""
    global wav2lip_animator
    if wav2lip_animator is None:
        wav2lip_animator = Wav2LipAnimator()
    return wav2lip_animator