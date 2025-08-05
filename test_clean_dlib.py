#!/usr/bin/env python3
"""
Test script for clean dlib animator
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from clean_dlib_animator import CleanDlibAnimator

def test_clean_dlib():
    """Test clean dlib animator functionality"""
    
    print("🎭 Testing Clean Dlib Animator")
    print("=" * 50)
    
    # Initialize clean dlib animator
    try:
        animator = CleanDlibAnimator()
        print("✅ Clean dlib animator initialized")
    except Exception as e:
        print(f"❌ Failed to initialize clean dlib animator: {e}")
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
    
    # Generate test audio
    sample_rate = 22050
    duration = 3.0  # 3 seconds
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
    
    print("✅ Test audio created")
    
    # Test face generation
    print("🎬 Testing face generation...")
    
    for i in range(10):  # Test 10 frames
        # Vary the audio intensity
        test_audio = audio_data[i*1000:(i+1)*1000] if len(audio_data) > (i+1)*1000 else audio_data[:1000]
        
        face = animator.generate_face_for_audio_chunk(test_audio)
        
        if face is not None and face.shape[0] > 0:
            print(f"✅ Frame {i+1}: Generated face {face.shape}")
        else:
            print(f"❌ Frame {i+1}: Failed to generate face")
            return False
    
    print("✅ All frames generated successfully!")
    return True

if __name__ == "__main__":
    success = test_clean_dlib()
    if success:
        print("\n🎉 Clean dlib test completed successfully!")
        print("🎭 Your Madame Leota can now use clean dlib animation!")
    else:
        print("\n❌ Clean dlib test failed")
        print("📝 Check the error messages above for troubleshooting") 