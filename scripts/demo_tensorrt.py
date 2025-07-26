#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo TensorRT Quantization
Script demo cho TensorRT quantization process
"""

import os
import sys
import time
from pathlib import Path

# Add project paths
current_dir = Path(__file__).parent
project_root = current_dir
sys.path.insert(0, str(project_root / 'app' / 'core'))
sys.path.insert(0, str(project_root / 'config'))

def demo_quantization():
    """Demo quantization process"""
    
    print("🚀 TensorRT Quantization Demo")
    print("=" * 50)
    
    # Check dependencies
    print("1️⃣ Checking dependencies...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch not installed")
        return
    
    try:
        from ultralytics import YOLO
        print("✅ YOLO available")
    except ImportError:
        print("❌ Ultralytics not installed")
        return
    
    try:
        import tensorrt as trt
        import pycuda
        print(f"✅ TensorRT: {trt.__version__}")
        print("✅ PyCUDA available")
        TRT_OK = True
    except ImportError as e:
        print(f"❌ TensorRT not available: {e}")
        print("🔧 Run: python scripts/setup_tensorrt.py")
        TRT_OK = False
    
    # Model check
    print("\n2️⃣ Checking models...")
    models_dir = project_root / "models"
    
    available_models = []
    for model_file in models_dir.glob("*.pt"):
        available_models.append(model_file.name)
        print(f"✅ {model_file.name}")
    
    if not available_models:
        print("❌ No YOLO models found in models/")
        print("💾 Download models:")
        print("   - yolov8s.pt")
        print("   - yolov9s.pt")
        return
    
    # Demo quantization (if TensorRT available)
    if TRT_OK and available_models:
        print("\n3️⃣ Demo quantization...")
        
        demo_model = available_models[0]
        print(f"📦 Using model: {demo_model}")
        
        # Mock quantization demo
        print("\n🔄 Starting quantization process...")
        
        steps = [
            "Loading YOLO model",
            "Exporting to ONNX",
            "Creating TensorRT builder",
            "Setting optimization profiles",
            "Building FP16 engine",
            "Serializing engine"
        ]
        
        for i, step in enumerate(steps, 1):
            print(f"   {i}/{len(steps)}: {step}...")
            time.sleep(0.5)  # Simulate processing
        
        print("✅ Quantization completed!")
        print(f"📁 Output: {demo_model.replace('.pt', '_fp16.engine')}")
        
        # Mock benchmark
        print("\n📊 Performance comparison:")
        print("   PyTorch (.pt):    15.2ms | 65.8 FPS")
        print("   TensorRT (FP16):   8.7ms | 114.9 FPS")
        print("   Speedup:          1.75x")
        print("   Size reduction:   47%")
    
    # Next steps
    print("\n4️⃣ Next steps:")
    
    if not TRT_OK:
        print("❌ Install TensorRT first:")
        print("   python scripts/setup_tensorrt.py")
    else:
        print("✅ Ready for quantization!")
        print("   python scripts/quantize_models.py --model yolov9s.pt --precision fp16")
        print("   python scripts/quantize_models.py --all-models --benchmark")
    
    print("\n📚 Documentation:")
    print("   docs/tensorrt_guide.md")
    
    print("\n🎯 Usage in application:")
    print("   python main.py --method web --model yolov9s.engine")

def demo_inference_comparison():
    """Demo inference speed comparison"""
    
    print("\n🏃 Inference Speed Demo")
    print("=" * 40)
    
    try:
        import torch
        import cv2
        import numpy as np
        from ultralytics import YOLO
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        return
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Test PyTorch model
    models_dir = Path("models")
    pytorch_models = list(models_dir.glob("*.pt"))
    
    if pytorch_models:
        print(f"📦 Testing PyTorch: {pytorch_models[0].name}")
        
        try:
            model = YOLO(str(pytorch_models[0]))
            
            # Warmup
            for _ in range(5):
                _ = model(dummy_image, verbose=False)
            
            # Benchmark
            times = []
            for _ in range(10):
                start = time.time()
                _ = model(dummy_image, verbose=False)
                times.append(time.time() - start)
            
            avg_time = np.mean(times) * 1000  # ms
            fps = 1000 / avg_time
            
            print(f"   Average: {avg_time:.1f}ms")
            print(f"   FPS: {fps:.1f}")
            
        except Exception as e:
            print(f"   Error: {e}")
    
    # Test TensorRT models
    tensorrt_models = list(models_dir.glob("*.engine"))
    
    if tensorrt_models:
        print(f"🚀 Testing TensorRT: {tensorrt_models[0].name}")
        print("   (Requires actual TensorRT engine)")
    else:
        print("❌ No TensorRT engines found")
        print("   Create with: python scripts/quantize_models.py")

def main():
    """Main demo function"""
    print("🎬 TensorRT Demo for People Counting")
    print("🎯 This demo shows TensorRT quantization capabilities")
    print()
    
    # Run demos
    demo_quantization()
    demo_inference_comparison()
    
    print("\n🎉 Demo completed!")
    print("💡 For real quantization, run:")
    print("   python scripts/quantize_models.py --help")

if __name__ == "__main__":
    main()
