#!/usr/bin/env python3
"""Test CUDA context in web environment"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_cuda_context():
    try:
        print("🧪 Testing CUDA context initialization...")
        
        # Test 1: Basic CUDA
        import pycuda.driver as cuda
        cuda.init()
        print(f"✅ CUDA devices found: {cuda.Device.count()}")
        
        # Test 2: TensorRT inference
        from app.core.tensorrt_inference import TensorRTInference
        
        engine_path = "models/yolov9s_fp16.engine"
        if not os.path.exists(engine_path):
            print(f"❌ Engine not found: {engine_path}")
            return False
        
        print(f"🚀 Testing TensorRT engine: {engine_path}")
        trt_model = TensorRTInference(engine_path)
        print("✅ TensorRT engine loaded successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cuda_context()
    print(f"\n{'🎉 SUCCESS' if success else '❌ FAILED'}")
