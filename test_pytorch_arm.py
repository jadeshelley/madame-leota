#!/usr/bin/env python3
"""
PyTorch ARM Compatibility Test
This script tests if PyTorch can be installed and used on ARM architecture
"""

import sys
import platform
import subprocess
import os

def print_status(message):
    print(f"✅ {message}")

def print_error(message):
    print(f"❌ {message}")

def print_warning(message):
    print(f"⚠️ {message}")

def check_system_info():
    """Check system architecture and Python version"""
    print("🔍 Checking system information...")
    
    arch = platform.machine()
    system = platform.system()
    python_version = platform.python_version()
    
    print(f"Architecture: {arch}")
    print(f"System: {system}")
    print(f"Python version: {python_version}")
    
    if arch not in ['aarch64', 'armv7l', 'armv8l']:
        print_error(f"Unsupported architecture: {arch}")
        print_error("This script is for ARM architecture only")
        return False
    
    if system != 'Linux':
        print_warning(f"System {system} may have compatibility issues")
    
    return True

def check_memory():
    """Check available memory"""
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    memory_kb = int(line.split()[1])
                    memory_gb = memory_kb / (1024 * 1024)
                    print(f"Available memory: {memory_gb:.1f}GB")
                    
                    if memory_gb < 4:
                        print_warning("Less than 4GB RAM - PyTorch may be slow")
                    return memory_gb
    except:
        print_warning("Could not determine memory size")
        return None

def test_pytorch_installation():
    """Test PyTorch installation"""
    print("\n🧠 Testing PyTorch installation...")
    
    try:
        import torch
        print_status(f"PyTorch {torch.__version__} installed successfully")
        
        # Test basic operations
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = torch.mm(x, y)
        print_status("Basic tensor operations working")
        
        # Check CUDA
        if torch.cuda.is_available():
            print_status("CUDA available (unexpected on Pi)")
        else:
            print_status("CUDA not available (expected on Pi)")
        
        return True
        
    except ImportError as e:
        print_error(f"PyTorch not installed: {e}")
        return False
    except Exception as e:
        print_error(f"PyTorch test failed: {e}")
        return False

def test_pytorch_installation_methods():
    """Test different PyTorch installation methods"""
    print("\n🔧 Testing PyTorch installation methods...")
    
    arch = platform.machine()
    
    if arch == 'aarch64':
        print("Testing ARM64 PyTorch installation methods...")
        
        methods = [
            ("Official CPU wheel", "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"),
            ("Alternative index", "pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu"),
            ("No binary (build from source)", "pip install torch torchvision torchaudio --no-binary torch")
        ]
        
        for name, command in methods:
            print(f"\nTrying: {name}")
            print(f"Command: {command}")
            
            try:
                result = subprocess.run(command.split(), capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    print_status(f"{name} succeeded!")
                    return True
                else:
                    print_error(f"{name} failed:")
                    print(result.stderr)
            except subprocess.TimeoutExpired:
                print_error(f"{name} timed out (took too long)")
            except Exception as e:
                print_error(f"{name} error: {e}")
    
    elif arch in ['armv7l', 'armv8l']:
        print("Testing ARM32 PyTorch installation methods...")
        
        methods = [
            ("Legacy ARM32 wheel", "pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html"),
            ("Build from source", "pip install torch torchvision torchaudio --no-binary torch")
        ]
        
        for name, command in methods:
            print(f"\nTrying: {name}")
            print(f"Command: {command}")
            
            try:
                result = subprocess.run(command.split(), capture_output=True, text=True, timeout=600)
                if result.returncode == 0:
                    print_status(f"{name} succeeded!")
                    return True
                else:
                    print_error(f"{name} failed:")
                    print(result.stderr)
            except subprocess.TimeoutExpired:
                print_error(f"{name} timed out (took too long)")
            except Exception as e:
                print_error(f"{name} error: {e}")
    
    return False

def main():
    print("🧠 PyTorch ARM Compatibility Test")
    print("=" * 40)
    
    # Check system
    if not check_system_info():
        sys.exit(1)
    
    # Check memory
    memory = check_memory()
    
    # Test existing PyTorch installation
    if test_pytorch_installation():
        print_status("PyTorch is already working!")
        return
    
    # Try to install PyTorch
    print("\n📦 PyTorch not found, attempting installation...")
    
    if test_pytorch_installation_methods():
        print_status("PyTorch installation successful!")
        
        # Test the installation
        if test_pytorch_installation():
            print_status("🎉 PyTorch is now working on ARM!")
        else:
            print_error("PyTorch installation succeeded but test failed")
    else:
        print_error("❌ All PyTorch installation methods failed")
        print_error("PyTorch may not be compatible with this ARM setup")
        print_error("Consider using a different approach or hardware")

if __name__ == "__main__":
    main() 