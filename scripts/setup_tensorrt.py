#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorRT Setup và Installation Guide
"""

import sys
import subprocess
import platform
import pkg_resources

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 8:
        print("❌ TensorRT requires Python 3.8+")
        return False
    
    print("✅ Python version compatible")
    return True

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            print(f"🚀 CUDA version: {cuda_version}")
            print(f"📱 GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"   GPU {i}: {gpu_name}")
            return True
        else:
            print("❌ CUDA không khả dụng")
            return False
    except ImportError:
        print("❌ PyTorch không được cài đặt")
        return False

def check_installed_packages():
    """Check TensorRT related packages"""
    packages_to_check = [
        'tensorrt',
        'pycuda',
        'nvidia-tensorrt',
        'torch',
        'ultralytics'
    ]
    
    print("\n📦 Checking installed packages:")
    installed = {}
    
    for package in packages_to_check:
        try:
            version = pkg_resources.get_distribution(package).version
            print(f"✅ {package}: {version}")
            installed[package] = version
        except pkg_resources.DistributionNotFound:
            print(f"❌ {package}: Not installed")
            installed[package] = None
    
    return installed

def install_tensorrt():
    """Install TensorRT và dependencies"""
    print("\n🔧 Installing TensorRT và dependencies...")
    
    # CUDA toolkit check
    print("1️⃣ Checking CUDA toolkit...")
    if not check_cuda():
        print("   ⚠️ Cần cài đặt CUDA toolkit trước:")
        print("   💾 Download từ: https://developer.nvidia.com/cuda-downloads")
        return False
    
    # Install commands
    install_commands = [
        # PyCUDA
        ["pip", "install", "pycuda"],
        
        # TensorRT (từ pip)
        ["pip", "install", "nvidia-tensorrt"],
        
        # Hoặc specific version
        # ["pip", "install", "tensorrt==8.6.1"],
        
        # Additional dependencies
        ["pip", "install", "nvidia-cuda-runtime-cu12"],
        ["pip", "install", "nvidia-cuda-cupti-cu12"],
    ]
    
    for i, cmd in enumerate(install_commands, 1):
        print(f"\n{i}️⃣ Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ Success: {cmd[2]}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed: {e}")
            print(f"Error output: {e.stderr}")
            return False
    
    return True

def create_test_script():
    """Create TensorRT test script"""
    test_script = '''#!/usr/bin/env python3
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
        print(f"\\n🔍 Testing {name}...")
        results[name] = test_func()
    
    # Summary
    print("\\n📊 Test Results:")
    print("="*30)
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\\n🎉 All tests passed! TensorRT ready to use.")
    else:
        print("\\n❌ Some tests failed. Check installation.")
'''
    
    with open('test_tensorrt.py', 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    print("📝 Created test_tensorrt.py")

def print_installation_guide():
    """Print comprehensive installation guide"""
    guide = '''
🚀 TensorRT Installation Guide
=====================================

1️⃣ System Requirements:
   - Python 3.8+
   - NVIDIA GPU with compute capability 6.0+
   - CUDA 11.8+ hoặc 12.x
   - cuDNN 8.x

2️⃣ Pre-installation:
   - Cài đặt NVIDIA Driver mới nhất
   - Cài đặt CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
   - Cài đặt cuDNN: https://developer.nvidia.com/cudnn

3️⃣ Installation Options:

   Option A: pip install (Recommended)
   ```bash
   pip install nvidia-tensorrt
   pip install pycuda
   ```

   Option B: conda install
   ```bash
   conda install -c conda-forge tensorrt
   conda install -c conda-forge pycuda
   ```

   Option C: Manual installation
   - Download TensorRT từ NVIDIA
   - Extract và setup PYTHONPATH

4️⃣ Verification:
   ```bash
   python test_tensorrt.py
   ```

5️⃣ Troubleshooting:
   - Kiểm tra CUDA version compatibility
   - Ensure GPU driver supports CUDA version
   - Check environment variables:
     * CUDA_HOME
     * LD_LIBRARY_PATH (Linux)
     * PATH (Windows)

6️⃣ Performance Tips:
   - Use FP16 precision cho balance speed/accuracy
   - Use INT8 với calibration cho maximum speed
   - Set workspace size dựa trên GPU memory

📚 Resources:
   - TensorRT Developer Guide: https://docs.nvidia.com/deeplearning/tensorrt/
   - YOLO TensorRT: https://github.com/ultralytics/ultralytics
'''
    
    print(guide)

def main():
    """Main setup function"""
    print("🚀 TensorRT Setup for People Counting System")
    print("="*60)
    
    # System checks
    if not check_python_version():
        sys.exit(1)
    
    # Check current installations
    installed = check_installed_packages()
    
    # Check CUDA
    cuda_ok = check_cuda()
    
    # Installation recommendations
    print("\n🎯 Recommendations:")
    
    if installed.get('tensorrt') is None:
        print("❌ TensorRT not installed")
        
        response = input("\n🤔 Install TensorRT now? (y/n): ").lower().strip()
        if response == 'y':
            if install_tensorrt():
                print("✅ TensorRT installation completed")
            else:
                print("❌ TensorRT installation failed")
        else:
            print("⏭️ Skipping installation")
    else:
        print("✅ TensorRT already installed")
    
    # Create test script
    create_test_script()
    
    # Print guide
    print_installation_guide()
    
    print("\n🎉 Setup completed!")
    print("Run 'python test_tensorrt.py' để test installation")

if __name__ == "__main__":
    main()
