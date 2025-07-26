#!/usr/bin/env python3
"""Test TensorRT installation"""

def test_tensorrt():
    """Test TensorRT imports và functionality"""
    try:
        import tensorrt as trt
        print(f"✅ TensorRT version: {trt.__version__}")
        
        # Test logger
        logger = trt.Logger(trt.Logger.WARNING)
        print("✅ TensorRT Logger OK")
        
        # Test builder
        builder = trt.Builder(logger)
        print("✅ TensorRT Builder OK")
        
        return True
    except ImportError as e:
        print(f"❌ TensorRT import failed: {e}")
        return False

def test_pycuda():
    """Test PyCUDA"""
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        print("✅ PyCUDA OK")
        
        # Get device info
        device = cuda.Device(0)
        print(f"✅ GPU: {device.name()}")
        
        return True
    except ImportError as e:
        print(f"❌ PyCUDA import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ PyCUDA error: {e}")
        return False

def test_torch_cuda():
    """Test PyTorch CUDA"""
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.version.cuda}")
            print(f"✅ GPU count: {torch.cuda.device_count()}")
            return True
        else:
            print("❌ CUDA not available in PyTorch")
            return False
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing TensorRT Installation")
    print("="*50)
    
    # Test all components
    tests = [
        ("PyTorch CUDA", test_torch_cuda),
        ("PyCUDA", test_pycuda), 
        ("TensorRT", test_tensorrt)
    ]
    
    results = {}
    for name, test_func in tests:
        print(f"\n🔍 Testing {name}...")
        results[name] = test_func()
    
    # Summary
    print("\n📊 Test Results:")
    print("="*30)
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! TensorRT ready to use.")
    else:
        print("\n❌ Some tests failed. Check installation.")
