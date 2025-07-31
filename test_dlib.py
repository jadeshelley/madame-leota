#!/usr/bin/env python3
"""
Test script to diagnose dlib installation on Raspberry Pi
"""

print("🔍 Testing dlib installation...")

# Test 1: Basic import
try:
    import dlib
    print("✅ dlib imported successfully")
    print(f"📍 dlib version: {dlib.version}")
except ImportError as e:
    print(f"❌ dlib import failed: {e}")
    print("💡 Solution: pip install dlib")
    exit(1)
except Exception as e:
    print(f"❌ dlib import error: {e}")
    exit(1)

# Test 2: Face detector
try:
    detector = dlib.get_frontal_face_detector()
    print("✅ Face detector created")
except Exception as e:
    print(f"❌ Face detector failed: {e}")
    exit(1)

# Test 3: Shape predictor (this will fail initially - that's OK)
try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    print("✅ Shape predictor loaded (already downloaded)")
except Exception as e:
    print(f"⚠️ Shape predictor not found (will auto-download): {e}")

# Test 4: OpenCV + imutils
try:
    import cv2
    print(f"✅ OpenCV version: {cv2.__version__}")
except ImportError as e:
    print(f"❌ OpenCV import failed: {e}")

try:
    import imutils
    print("✅ imutils imported")
except ImportError as e:
    print(f"❌ imutils import failed: {e}")
    print("💡 Solution: pip install imutils")

# Test 5: Try to create DlibFaceAnimator
try:
    print("\n🎭 Testing DlibFaceAnimator...")
    import sys
    sys.path.append('src')
    from dlib_face_animator import DlibFaceAnimator
    
    animator = DlibFaceAnimator()
    print("✅ DlibFaceAnimator created successfully")
    
except Exception as e:
    print(f"❌ DlibFaceAnimator failed: {e}")
    import traceback
    print(f"📋 Full traceback:\n{traceback.format_exc()}")

print("\n🎯 dlib test complete!") 