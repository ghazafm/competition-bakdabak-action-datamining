#!/usr/bin/env python3
"""
GPU and CUDA diagnostic script for YOLOv11 training.
Run this to diagnose GPU/CUDA issues before training.
"""

import sys

print("=" * 60)
print("GPU/CUDA Diagnostic Tool")
print("=" * 60)

# Check Python version
print(f"\n1. Python Version: {sys.version}")
if not sys.version.startswith('3.10'):
    print("   ⚠️  WARNING: This project requires Python 3.10.x")
else:
    print("   ✅ Python version OK")

# Check PyTorch
try:
    import torch
    print(f"\n2. PyTorch Version: {torch.__version__}")
    print(f"   CUDA compiled version: {torch.version.cuda if torch.version.cuda else 'Not compiled with CUDA'}")
except ImportError:
    print("\n2. ❌ PyTorch not installed!")
    print("   Install with: pip install -e .")
    sys.exit(1)

# Check CUDA availability
print(f"\n3. CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   ✅ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA runtime version: {torch.version.cuda}")
    print(f"   Number of GPUs: {torch.cuda.device_count()}")
    print(f"   Current GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Try a simple CUDA operation
    try:
        x = torch.rand(3, 3).cuda()
        print("   ✅ CUDA operations working")
    except Exception as e:
        print(f"   ❌ CUDA operation failed: {e}")
        print("   This usually means CUDA driver version mismatch")
else:
    print("   ⚠️  No CUDA GPU detected")
    
    # Check MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        print(f"\n4. MPS (Apple Silicon) Available: True")
        print("   ✅ Will use MPS acceleration")
    else:
        print(f"\n4. MPS (Apple Silicon) Available: False")
        print("   ⚠️  Will use CPU (training will be slower)")

# Check other dependencies
print("\n5. Checking other dependencies...")
try:
    from ultralytics import YOLO
    print("   ✅ ultralytics installed")
except ImportError:
    print("   ❌ ultralytics not installed")

try:
    from roboflow import Roboflow
    print("   ✅ roboflow installed")
except ImportError:
    print("   ❌ roboflow not installed")

try:
    from dotenv import load_dotenv
    print("   ✅ python-dotenv installed")
except ImportError:
    print("   ❌ python-dotenv not installed")

# Final recommendations
print("\n" + "=" * 60)
print("RECOMMENDATIONS")
print("=" * 60)

if not torch.cuda.is_available():
    print("\n⚠️  CUDA not available. For your NVIDIA A100 GPU:")
    print("\n   Run these commands to fix CUDA:")
    print("   pip uninstall torch torchvision torchaudio")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116")
    print("\n   Or force CPU training by setting in .env:")
    print("   DEVICE=cpu")
    print("   BATCH_SIZE=4")
else:
    print("\n✅ GPU is properly configured!")
    print("   Recommended settings for A100 GPU:")
    print("   DEVICE=cuda")
    print("   BATCH_SIZE=16  # or higher")
    print("   EPOCHS=10")

print("\n" + "=" * 60)
