#!/usr/bin/env python3
"""
Test script for D-ID API integration
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from did_animator import DIDAnimator

def test_did_api():
    """Test D-ID API functionality"""
    
    print("🎭 Testing D-ID API Integration")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv('DID_API_KEY')
    if not api_key:
        print("❌ No D-ID API key found!")
        print("📝 To get a free API key:")
        print("   1. Go to: https://studio.d-id.com/")
        print("   2. Sign up for a free account")
        print("   3. Get your API key from the dashboard")
        print("   4. Set environment variable: export DID_API_KEY='your_key_here'")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...")
    
    # Initialize D-ID animator
    try:
        animator = DIDAnimator(api_key)
        print("✅ D-ID animator initialized")
    except Exception as e:
        print(f"❌ Failed to initialize D-ID animator: {e}")
        return False
    
    # Load base face
    face_path = Path("assets/faces/realistic_face.jpg")
    if not face_path.exists():
        print(f"❌ Face image not found: {face_path}")
        print("📝 Please add a face image to assets/faces/realistic_face.jpg")
        return False
    
    success = animator.load_base_face(str(face_path))
    if not success:
        print("❌ Failed to load base face")
        return False
    
    print("✅ Base face loaded successfully")
    
    # Create test audio (simple sine wave)
    import numpy as np
    import wave
    
    # Generate test audio
    sample_rate = 22050
    duration = 3.0  # 3 seconds
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
    
    # Convert to 16-bit PCM
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Save as WAV file
    with wave.open("test_audio.wav", "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    print("✅ Test audio created: test_audio.wav")
    
    # Read audio data
    with open("test_audio.wav", "rb") as f:
        audio_bytes = f.read()
    
    print("🎬 Creating talking face video...")
    print("⏳ This may take 30-60 seconds...")
    
    # Create talking video
    output_path = "test_talking_face.mp4"
    result = animator.create_talking_video(audio_bytes, output_path)
    
    if result:
        print(f"✅ Success! Video created: {output_path}")
        print("🎭 You can now play the video to see the talking face!")
        return True
    else:
        print("❌ Failed to create talking video")
        return False

if __name__ == "__main__":
    success = test_did_api()
    if success:
        print("\n🎉 D-ID API test completed successfully!")
        print("🎭 Your Madame Leota can now use professional-quality lip sync!")
    else:
        print("\n❌ D-ID API test failed")
        print("📝 Check the error messages above for troubleshooting") 