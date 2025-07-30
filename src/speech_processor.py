"""
Speech Processor for Madame Leota
Handles speech recognition and text-to-speech with phoneme mapping for lip sync
"""

import asyncio
import logging
import speech_recognition as sr
import pyttsx3
import numpy as np
from typing import List, Tuple, Optional
import io
import wave
import threading
from config import *

class SpeechProcessor:
    def __init__(self, audio_manager):
        self.logger = logging.getLogger(__name__)
        self.audio_manager = audio_manager
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Initialize text-to-speech
        self.tts_engine = pyttsx3.init()
        self._setup_tts()
        
        # Phoneme mapping for lip sync
        self.phoneme_map = self._create_phoneme_map()
        
        # Calibrate microphone
        self._calibrate_microphone()
    
    def _setup_tts(self):
        """Configure the TTS engine"""
        try:
            voices = self.tts_engine.getProperty('voices')
            if voices and len(voices) > TTS_VOICE_INDEX:
                self.tts_engine.setProperty('voice', voices[TTS_VOICE_INDEX].id)
            
            self.tts_engine.setProperty('rate', TTS_RATE)
            self.tts_engine.setProperty('volume', 0.9)
            
        except Exception as e:
            self.logger.warning(f"TTS setup warning: {e}")
    
    def _calibrate_microphone(self):
        """Calibrate microphone for ambient noise"""
        try:
            with self.microphone as source:
                self.logger.info("Calibrating microphone for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
                self.logger.info("Microphone calibrated")
        except Exception as e:
            self.logger.error(f"Microphone calibration failed: {e}")
    
    def _create_phoneme_map(self) -> dict:
        """Create mapping of phonemes to mouth shapes for animation"""
        return {
            # Vowels
            'AA': 'mouth_open',     # father
            'AE': 'mouth_wide',     # cat
            'AH': 'mouth_open',     # cut
            'AO': 'mouth_round',    # dog
            'AW': 'mouth_round',    # how
            'AY': 'mouth_wide',     # my
            'EH': 'mouth_wide',     # bet
            'ER': 'mouth_narrow',   # bird
            'EY': 'mouth_wide',     # bait
            'IH': 'mouth_narrow',   # bit
            'IY': 'mouth_narrow',   # beat
            'OW': 'mouth_round',    # boat
            'OY': 'mouth_round',    # boy
            'UH': 'mouth_narrow',   # book
            'UW': 'mouth_round',    # boot
            
            # Consonants
            'B': 'mouth_closed',    # b
            'CH': 'mouth_narrow',   # ch
            'D': 'mouth_narrow',    # d
            'DH': 'mouth_narrow',   # th (the)
            'F': 'mouth_narrow',    # f
            'G': 'mouth_open',      # g
            'HH': 'mouth_open',     # h
            'JH': 'mouth_narrow',   # j
            'K': 'mouth_open',      # k
            'L': 'mouth_narrow',    # l
            'M': 'mouth_closed',    # m
            'N': 'mouth_narrow',    # n
            'NG': 'mouth_narrow',   # ng
            'P': 'mouth_closed',    # p
            'R': 'mouth_narrow',    # r
            'S': 'mouth_narrow',    # s
            'SH': 'mouth_narrow',   # sh
            'T': 'mouth_narrow',    # t
            'TH': 'mouth_narrow',   # th (think)
            'V': 'mouth_narrow',    # v
            'W': 'mouth_round',     # w
            'Y': 'mouth_narrow',    # y
            'Z': 'mouth_narrow',    # z
            'ZH': 'mouth_narrow',   # zh
            
            # Silence
            'SIL': 'mouth_closed',
            'sp': 'mouth_closed'
        }
    
    async def listen_for_speech(self) -> Optional[str]:
        """Listen for user speech and convert to text"""
        try:
            self.logger.debug("Listening for speech...")
            
            # Use asyncio to run the blocking operation in a thread
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, self._recognize_speech)
            
            if text:
                self.logger.info(f"Recognized: {text}")
                return text.lower().strip()
            
            return None
            
        except Exception as e:
            self.logger.error(f"Speech recognition error: {e}")
            return None
    
    def _recognize_speech(self) -> Optional[str]:
        """Blocking speech recognition"""
        try:
            with self.microphone as source:
                # Listen for audio
                audio = self.recognizer.listen(
                    source, 
                    timeout=RECOGNITION_TIMEOUT,
                    phrase_time_limit=RECOGNITION_PHRASE_TIMEOUT
                )
            
            # Recognize speech using Google's service
            text = self.recognizer.recognize_google(audio)
            return text
            
        except sr.WaitTimeoutError:
            self.logger.debug("No speech detected (timeout)")
            return None
        except sr.UnknownValueError:
            self.logger.debug("Could not understand audio")
            return None
        except sr.RequestError as e:
            self.logger.error(f"Speech recognition service error: {e}")
            return None
    
    async def text_to_speech_with_phonemes(self, text: str) -> Tuple[bytes, List[dict]]:
        """Convert text to speech and generate phoneme timing data"""
        try:
            # Generate audio
            audio_data = await self._generate_audio(text)
            
            # Generate phoneme timing (simplified approach)
            phonemes = self._generate_phoneme_timing(text, len(audio_data))
            
            return audio_data, phonemes
            
        except Exception as e:
            self.logger.error(f"TTS error: {e}")
            return b'', []
    
    async def _generate_audio(self, text: str) -> bytes:
        """Generate audio data from text"""
        try:
            # Create a temporary file in memory
            audio_buffer = io.BytesIO()
            
            # Configure TTS to save to buffer
            temp_file = f"{AUDIO_CACHE_DIR}/temp_speech.wav"
            
            # Run TTS in thread to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._save_tts_audio, text, temp_file)
            
            # Read the audio file
            with open(temp_file, 'rb') as f:
                audio_data = f.read()
            
            return audio_data
            
        except Exception as e:
            self.logger.error(f"Audio generation error: {e}")
            return b''
    
    def _save_tts_audio(self, text: str, filename: str):
        """Save TTS audio to file (blocking operation)"""
        self.tts_engine.save_to_file(text, filename)
        self.tts_engine.runAndWait()
    
    def _generate_phoneme_timing(self, text: str, audio_length: int) -> List[dict]:
        """Generate simplified phoneme timing for lip sync"""
        words = text.split()
        phonemes = []
        
        if not words:
            return phonemes
        
        # Estimate timing (very simplified approach)
        word_duration = audio_length / len(words) if words else 0
        current_time = 0
        
        for word in words:
            # Simple phoneme mapping based on letters (very basic)
            word_phonemes = self._word_to_phonemes(word)
            phoneme_duration = word_duration / len(word_phonemes) if word_phonemes else 0
            
            for phoneme in word_phonemes:
                mouth_shape = self.phoneme_map.get(phoneme, 'mouth_neutral')
                phonemes.append({
                    'phoneme': phoneme,
                    'start_time': current_time,
                    'duration': phoneme_duration,
                    'mouth_shape': mouth_shape
                })
                current_time += phoneme_duration
            
            # Add brief pause between words
            phonemes.append({
                'phoneme': 'SIL',
                'start_time': current_time,
                'duration': phoneme_duration * 0.2,
                'mouth_shape': 'mouth_closed'
            })
            current_time += phoneme_duration * 0.2
        
        return phonemes
    
    def _word_to_phonemes(self, word: str) -> List[str]:
        """Convert word to phonemes (very simplified)"""
        # This is a very basic approximation
        # In a real implementation, you'd use a proper phoneme dictionary
        phonemes = []
        
        vowels = 'aeiou'
        consonants = 'bcdfghjklmnpqrstvwxyz'
        
        for char in word.lower():
            if char in vowels:
                if char == 'a':
                    phonemes.append('AE')
                elif char == 'e':
                    phonemes.append('EH')
                elif char == 'i':
                    phonemes.append('IH')
                elif char == 'o':
                    phonemes.append('AO')
                elif char == 'u':
                    phonemes.append('UH')
            elif char in consonants:
                phonemes.append(char.upper())
        
        return phonemes if phonemes else ['SIL'] 