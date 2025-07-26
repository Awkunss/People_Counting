#!/usr/bin/env python3
"""
Benchmark Models Performance Script
So sánh performance giữa PyTorch và TensorRT models
"""

import sys
import os
import time
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO
from app.core.tensorrt_inference import TensorRTInference

def benchmark_pytorch_model(model_path, num_runs=100):
    """Benchmark PyTorch YOLO model"""
    print(f"\n📊 Benchmarking PyTorch: {model_path}")
    
    # Load model
    model = YOLO(model_path)
    
    # Create dummy input
    input_tensor = torch.randn(1, 3, 640, 640)
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
        model.model = model.model.cuda()
    
    # Warmup
    print("Warming up...")
    for _ in range(10):
        with torch.no_grad():
            _ = model.model(input_tensor)
    
    # Benchmark
    print(f"Running {num_runs} iterations...")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model.model(input_tensor)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = (total_time / num_runs) * 1000  # ms
    fps = num_runs / total_time
    
    print(f"✅ PyTorch Results:")
    print(f"   Average time: {avg_time:.2f} ms")
    print(f"   FPS: {fps:.1f}")
    
    return avg_time, fps

def benchmark_tensorrt_engine(engine_path, num_runs=100):
    """Benchmark TensorRT engine"""
    print(f"\n🚀 Benchmarking TensorRT: {engine_path}")
    
    if not os.path.exists(engine_path):
        print(f"❌ Engine not found: {engine_path}")
        return None, None
    
    try:
        # Load TensorRT engine
        trt_inference = TensorRTInference(engine_path)
        
        # Create dummy input
        dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
        
        # Warmup
        print("Warming up...")
        for _ in range(10):
            _ = trt_inference._inference(dummy_input)
        
        # Benchmark
        print(f"Running {num_runs} iterations...")
        
        start_time = time.time()
        for _ in range(num_runs):
            _ = trt_inference._inference(dummy_input)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = (total_time / num_runs) * 1000  # ms
        fps = num_runs / total_time
        
        print(f"✅ TensorRT Results:")
        print(f"   Average time: {avg_time:.2f} ms")
        print(f"   FPS: {fps:.1f}")
        
        return avg_time, fps
        
    except Exception as e:
        print(f"❌ TensorRT benchmark failed: {e}")
        return None, None

def main():
    print("🔥 MODEL PERFORMANCE BENCHMARK")
    print("=" * 50)
    
    # Define models to test
    models_dir = project_root / "models"
    pytorch_model = models_dir / "yolov9s.pt"
    tensorrt_engine = models_dir / "yolov9s_fp16.engine"
    
    num_runs = 50  # Reduced for faster testing
    
    results = {}
    
    # Test PyTorch model
    if pytorch_model.exists():
        pt_time, pt_fps = benchmark_pytorch_model(str(pytorch_model), num_runs)
        results['pytorch'] = {'time': pt_time, 'fps': pt_fps}
    else:
        print(f"❌ PyTorch model not found: {pytorch_model}")
    
    # Test TensorRT engine
    if tensorrt_engine.exists():
        trt_time, trt_fps = benchmark_tensorrt_engine(str(tensorrt_engine), num_runs)
        results['tensorrt'] = {'time': trt_time, 'fps': trt_fps}
    else:
        print(f"❌ TensorRT engine not found: {tensorrt_engine}")
    
    # Compare results
    print("\n" + "=" * 50)
    print("📈 PERFORMANCE COMPARISON")
    print("=" * 50)
    
    if 'pytorch' in results and 'tensorrt' in results and results['tensorrt']['time'] is not None:
        pt_time = results['pytorch']['time']
        pt_fps = results['pytorch']['fps']
        trt_time = results['tensorrt']['time']
        trt_fps = results['tensorrt']['fps']
        
        speedup = pt_time / trt_time if trt_time else 0
        fps_improvement = (trt_fps / pt_fps - 1) * 100 if pt_fps else 0
        
        print(f"PyTorch:  {pt_time:.2f} ms ({pt_fps:.1f} FPS)")
        print(f"TensorRT: {trt_time:.2f} ms ({trt_fps:.1f} FPS)")
        print(f"")
        print(f"🚀 Speedup: {speedup:.2f}x")
        print(f"📊 FPS improvement: {fps_improvement:+.1f}%")
        
        if speedup > 1.5:
            print("🎉 Excellent TensorRT optimization!")
        elif speedup > 1.2:
            print("✅ Good TensorRT optimization!")
        else:
            print("⚠️  Modest TensorRT optimization")
    
    elif 'pytorch' in results:
        pt_time = results['pytorch']['time']
        pt_fps = results['pytorch']['fps']
        print(f"PyTorch:  {pt_time:.2f} ms ({pt_fps:.1f} FPS)")
        print(f"TensorRT: ❌ Failed to benchmark")
        print("\n⚠️  TensorRT benchmark failed - check engine compatibility")
    
    else:
        print("❌ No successful benchmarks to compare")
    
    print("\n💡 Note: Run with GPU for best TensorRT performance")

if __name__ == "__main__":
    main()
