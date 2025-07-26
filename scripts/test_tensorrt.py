#!/usr/bin/env python3
"""Simple test TensorRT engine"""

import sys
import os
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_tensorrt_engine():
    try:
        from app.core.tensorrt_inference import TensorRTInference
        
        engine_path = "models/yolov9s_fp16.engine"
        print(f"Testing TensorRT engine: {engine_path}")
        
        if not os.path.exists(engine_path):
            print(f"❌ Engine not found: {engine_path}")
            return False
        
        # Load engine
        print("Loading TensorRT engine...")
        trt_model = TensorRTInference(engine_path)
        print("✅ Engine loaded successfully!")
        
        # Test inference
        print("Testing inference...")
        dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
        
        result = trt_model._inference(dummy_input)
        print(f"✅ Inference successful! Output shape: {result.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 TENSORRT ENGINE TEST")
    print("=" * 30)
    
    success = test_tensorrt_engine()
    
    if success:
        print("\n🎉 TensorRT engine is working correctly!")
    else:
        print("\n❌ TensorRT engine test failed!")
        sys.exit(1)
