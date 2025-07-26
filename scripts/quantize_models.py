#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantize YOLO Models Script
Script để quantize YOLO models sang TensorRT engines
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project paths
current_dir = Path(__file__).parent
project_root = current_dir.parent  # Go up one level to project root
sys.path.insert(0, str(project_root / 'app' / 'core'))
sys.path.insert(0, str(project_root / 'config'))

# Default values for fallback
DEFAULT_TENSORRT_PRECISION = "fp16"
DEFAULT_TENSORRT_WORKSPACE_SIZE = 1024

try:
    from tensorrt_quantizer import TensorRTQuantizer, create_calibration_data
    from settings import MODELS_DIR, DATA_DIR, TENSORRT_PRECISION, TENSORRT_WORKSPACE_SIZE
    TRT_AVAILABLE = True
except ImportError as e:
    TRT_AVAILABLE = False
    # Use default values
    TENSORRT_PRECISION = DEFAULT_TENSORRT_PRECISION
    TENSORRT_WORKSPACE_SIZE = DEFAULT_TENSORRT_WORKSPACE_SIZE
    
    # Try to get basic paths
    try:
        from settings import MODELS_DIR, DATA_DIR
    except ImportError:
        MODELS_DIR = project_root / "models"
        DATA_DIR = project_root / "data"
    
    print(f"❌ TensorRT không khả dụng: {e}")
    print("Cài đặt TensorRT và dependencies:")
    print("pip install nvidia-tensorrt")
    print("pip install pycuda")

def setup_logging():
    """Setup logging"""
    import sys
    
    # Fix encoding issue trên Windows
    if sys.platform == 'win32':
        # Use UTF-8 encoding và remove emoji cho Windows console
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('quantization.log', encoding='utf-8')
            ]
        )
        # Set console to handle UTF-8
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('quantization.log', encoding='utf-8')
            ]
        )
    
    return logging.getLogger(__name__)

def quantize_model(model_name, precision='fp16', calibration_video=None, 
                  workspace_size=1024, benchmark=False):
    """Quantize một YOLO model"""
    
    logger = setup_logging()
    
    if not TRT_AVAILABLE:
        logger.error("❌ TensorRT không khả dụng")
        return False
    
    # Paths
    model_path = MODELS_DIR / model_name
    engine_name = model_name.replace('.pt', f'_{precision}.engine')
    engine_path = MODELS_DIR / engine_name
    
    logger.info(f"Quantizing {model_name} -> {engine_name}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Engine path: {engine_path}")
    
    # Check model exists
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return False
    
    # Create quantizer
    quantizer = TensorRTQuantizer(logger)
    
    # Calibration data cho INT8
    calibration_data = None
    if precision == 'int8':
        if calibration_video:
            calib_video_path = DATA_DIR / calibration_video
            if calib_video_path.exists():
                logger.info(f"Creating calibration data from: {calib_video_path}")
                calibration_data = create_calibration_data(str(calib_video_path), num_frames=100)
            else:
                logger.warning(f"Calibration video not found: {calib_video_path}")
                logger.info("Using random calibration data")
        else:
            logger.warning("INT8 precision needs calibration data")
            logger.info("Using FP16 instead")
            precision = 'fp16'
    
    # Quantize
    success = quantizer.quantize_model(
        model_path=str(model_path),
        engine_path=str(engine_path),
        precision=precision,
        calibration_data=calibration_data,
        workspace_size=workspace_size * 1024 * 1024  # Convert MB to bytes
    )
    
    if success:
        logger.info(f"Quantization completed: {engine_path}")
        
        # File size comparison
        model_size = model_path.stat().st_size / (1024 * 1024)  # MB
        engine_size = engine_path.stat().st_size / (1024 * 1024)  # MB
        compression_ratio = model_size / engine_size if engine_size > 0 else 0
        
        logger.info(f"Model size: {model_size:.2f} MB")
        logger.info(f"Engine size: {engine_size:.2f} MB")
        logger.info(f"Compression ratio: {compression_ratio:.2f}x")
        
        # Benchmark if requested
        if benchmark:
            logger.info("Running benchmark...")
            results = quantizer.benchmark_engine(str(engine_path))
            if results:
                logger.info(f"Performance:")
                logger.info(f"   Average inference: {results['avg_inference_time_ms']:.2f} ms")
                logger.info(f"   FPS: {results['fps']:.1f}")
                logger.info(f"   Total runs: {results['num_runs']}")
        
        return True
    else:
        logger.error("Quantization failed")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='YOLO TensorRT Quantization Tool')
    
    # Model selection
    parser.add_argument('--model', default='yolov9s.pt', 
                       help='YOLO model to quantize (default: yolov9s.pt)')
    
    # Precision
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'int8'], 
                       default=TENSORRT_PRECISION,
                       help=f'Quantization precision (default: {TENSORRT_PRECISION})')
    
    # Calibration
    parser.add_argument('--calibration-video', default='Test.mp4',
                       help='Video for INT8 calibration (default: Test.mp4)')
    
    # Workspace
    parser.add_argument('--workspace', type=int, default=TENSORRT_WORKSPACE_SIZE,
                       help=f'TensorRT workspace size in MB (default: {TENSORRT_WORKSPACE_SIZE})')
    
    # Benchmark
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark after quantization')
    
    # Batch quantization
    parser.add_argument('--all-models', action='store_true',
                       help='Quantize all available YOLO models')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    if not TRT_AVAILABLE:
        logger.error("❌ TensorRT không khả dụng")
        logger.info("Cài đặt dependencies:")
        logger.info("pip install nvidia-tensorrt")
        logger.info("pip install pycuda")
        sys.exit(1)
    
    # Single model hoặc all models
    if args.all_models:
        models_to_quantize = [
            'yolov8n.pt',
            'yolov8s.pt', 
            'yolov8m.pt',
            'yolov8l.pt',
            'yolov9s.pt'
        ]
        
        logger.info(f"Quantizing {len(models_to_quantize)} models...")
        
        success_count = 0
        for model in models_to_quantize:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing: {model}")
            logger.info(f"{'='*50}")
            
            success = quantize_model(
                model_name=model,
                precision=args.precision,
                calibration_video=args.calibration_video,
                workspace_size=args.workspace,
                benchmark=args.benchmark
            )
            
            if success:
                success_count += 1
            
            logger.info(f"Progress: {success_count}/{len(models_to_quantize)} completed")
        
        logger.info(f"\nBatch quantization completed: {success_count}/{len(models_to_quantize)} successful")
    
    else:
        # Single model
        success = quantize_model(
            model_name=args.model,
            precision=args.precision,
            calibration_video=args.calibration_video,
            workspace_size=args.workspace,
            benchmark=args.benchmark
        )
        
        if success:
            logger.info("Quantization completed successfully!")
        else:
            logger.error("Quantization failed!")
            sys.exit(1)

if __name__ == "__main__":
    main()
