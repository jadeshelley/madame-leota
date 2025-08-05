#!/usr/bin/env python3
"""
Test script for the clean mouth animation system
This will test if the system works before transferring to Pi
"""

import cv2
import numpy as np
import time
from pathlib import Path

def test_clean_mouth_animator():
    """Test the clean mouth animator"""
    print("🧪 Testing Clean Mouth Animator...")
    
    try:
        # Import the clean mouth animator
        from src.clean_mouth_animator import CleanMouthAnimator
        print("✅ Clean mouth animator imported successfully")
        
        # Create instance
        animator = CleanMouthAnimator()
        print("✅ Clean mouth animator instance created")
        
        # Test loading mouth shapes
        faces_dir = "assets/faces"
        success = animator.load_mouth_shapes(faces_dir)
        
        if success:
            print(f"✅ Loaded {len(animator.mouth_shapes)} mouth shapes")
            
            # Test generating a face with fake audio
            fake_audio = np.random.randn(1024).astype(np.float32) * 0.1
            face = animator.generate_face_for_audio_chunk(fake_audio)
            
            print(f"✅ Generated face with shape: {face.shape}")
            
            # Save test image
            cv2.imwrite("test_clean_mouth.png", face)
            print("✅ Saved test image as 'test_clean_mouth.png'")
            
            return True
        else:
            print("❌ Failed to load mouth shapes")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_face_animator():
    """Test the face animator"""
    print("\n🧪 Testing Face Animator...")
    
    try:
        # Mock display manager for testing
        class MockDisplayManager:
            def clear_screen(self):
                pass
            def display_image(self, image, position):
                pass
            def update_display(self):
                pass
            def get_screen_size(self):
                return (800, 600)
            def add_glow_effect(self, image, intensity):
                return image
        
        # Import face animator
        from src.face_animator import FaceAnimator
        print("✅ Face animator imported successfully")
        
        # Create instance
        display_manager = MockDisplayManager()
        animator = FaceAnimator(display_manager)
        print("✅ Face animator instance created")
        
        # Test if clean mouth animator is available
        if animator.clean_mouth_animator:
            print("✅ Clean mouth animator is available in face animator")
            return True
        else:
            print("❌ Clean mouth animator not available in face animator")
            return False
            
    except Exception as e:
        print(f"❌ Face animator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🎭 Testing Clean Animation System")
    print("=" * 50)
    
    # Test clean mouth animator
    clean_test = test_clean_mouth_animator()
    
    # Test face animator
    face_test = test_face_animator()
    
    print("\n" + "=" * 50)
    if clean_test and face_test:
        print("🎉 ALL TESTS PASSED! System is ready for Pi")
        print("\n📋 Next steps:")
        print("1. Transfer the updated code to your Raspberry Pi")
        print("2. Run: python3 main.py")
        print("3. The clean mouth animator should work with your existing mouth shape images")
    else:
        print("❌ Some tests failed. Check the errors above.")
    
    return clean_test and face_test

if __name__ == "__main__":
    main() 