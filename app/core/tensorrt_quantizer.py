#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorRT Quantization Module
Module hỗ trợ quantization và optimization TensorRT cho YOLO models
"""

import os
import sys
import logging
from pathlib import Path
import time

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    import numpy as np
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    logging.warning("TensorRT không khả dụng. Cài đặt TensorRT để sử dụng quantization.")

try:
    from ultralytics import YOLO
    import torch
    import cv2
except ImportError as e:
    logging.error(f"Thiếu dependencies: {e}")

class TensorRTQuantizer:
    """Class để quantize YOLO models sang TensorRT engine"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT không khả dụng. Cài đặt TensorRT trước.")
    
    def quantize_model(self, model_path, engine_path, precision='fp16', 
                      calibration_data=None, workspace_size=1<<30):
        """
        Quantize YOLO model thành TensorRT engine
        
        Args:
            model_path: Đường dẫn đến model YOLO (.pt)
            engine_path: Đường dẫn output engine (.engine)
            precision: 'fp32', 'fp16', hoặc 'int8'
            calibration_data: Data cho INT8 calibration
            workspace_size: TensorRT workspace size (default 1GB)
        """
        
        self.logger.info(f"Starting quantize {model_path} -> {engine_path}")
        self.logger.info(f"Precision: {precision}")
        
        try:
            # Load YOLO model
            model = YOLO(model_path)
            
            # Export sang ONNX trước
            onnx_path = str(engine_path).replace('.engine', '.onnx')
            self.logger.info(f"Export ONNX: {onnx_path}")
            
            # Export với output path cụ thể
            export_result = model.export(
                format='onnx',
                imgsz=640,
                dynamic=False,
                simplify=True,
                opset=11
            )
            
            # Kiểm tra file ONNX được tạo
            actual_onnx_path = export_result if isinstance(export_result, str) else onnx_path
            if not os.path.exists(actual_onnx_path):
                # Thử tìm file ONNX trong cùng thư mục với model
                model_dir = os.path.dirname(model_path)
                model_name = os.path.splitext(os.path.basename(model_path))[0]
                potential_onnx = os.path.join(model_dir, f"{model_name}.onnx")
                
                if os.path.exists(potential_onnx):
                    actual_onnx_path = potential_onnx
                    self.logger.info(f"Found ONNX at: {actual_onnx_path}")
                else:
                    raise FileNotFoundError(f"ONNX file not found. Expected: {onnx_path}")
            
            # Convert ONNX sang TensorRT
            self._onnx_to_tensorrt(
                onnx_path=actual_onnx_path,
                engine_path=engine_path,
                precision=precision,
                calibration_data=calibration_data,
                workspace_size=workspace_size
            )
            
            # Cleanup ONNX file
            if os.path.exists(actual_onnx_path):
                os.remove(actual_onnx_path)
                self.logger.info(f"Removed temp ONNX: {actual_onnx_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Quantization error: {str(e)}")
            return False
    
    def _onnx_to_tensorrt(self, onnx_path, engine_path, precision, 
                         calibration_data, workspace_size):
        """Convert ONNX sang TensorRT engine"""
        
        # Create builder và network
        builder = trt.Builder(self.trt_logger)
        config = builder.create_builder_config()
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        
        # Parse ONNX
        parser = trt.OnnxParser(network, self.trt_logger)
        
        self.logger.info(f"Parsing ONNX: {onnx_path}")
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    self.logger.error(f"ONNX Parse Error: {parser.get_error(i)}")
                raise RuntimeError("Failed to parse ONNX")
        
        # Config precision
        # Handle different TensorRT versions for workspace size
        try:
            # TensorRT 8.x and newer
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
        except AttributeError:
            # TensorRT 7.x and older  
            config.max_workspace_size = workspace_size
        
        if precision == 'fp16':
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                self.logger.info("FP16 optimization enabled")
            else:
                self.logger.warning("FP16 not supported, fallback to FP32")
        
        elif precision == 'int8':
            if builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                
                # INT8 calibrator
                if calibration_data:
                    calibrator = self._create_calibrator(calibration_data)
                    config.int8_calibrator = calibrator
                    self.logger.info("INT8 optimization with calibration enabled")
                else:
                    self.logger.warning("INT8 needs calibration data")
            else:
                self.logger.warning("INT8 not supported, fallback to FP32")
        
        # Optimization profiles
        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name
        input_shape = network.get_input(0).shape
        
        # Check if input has dynamic shape
        has_dynamic_shape = any(dim == -1 for dim in input_shape)
        
        if has_dynamic_shape:
            # Dynamic shapes cho batch size
            min_shape = (1, input_shape[1], input_shape[2], input_shape[3])
            opt_shape = (1, input_shape[1], input_shape[2], input_shape[3])
            max_shape = (4, input_shape[1], input_shape[2], input_shape[3])
            self.logger.info("Using dynamic batch size profile")
        else:
            # Static shape - tất cả dimensions phải giống nhau
            static_shape = tuple(input_shape)
            min_shape = opt_shape = max_shape = static_shape
            self.logger.info("Using static shape profile")
        
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        
        self.logger.info(f"Input shape: {input_shape}")
        self.logger.info(f"Optimization profile: min={min_shape}, opt={opt_shape}, max={max_shape}")
        
        # Build engine
        self.logger.info("Building TensorRT engine...")
        start_time = time.time()
        
        # Handle different TensorRT versions for engine building
        try:
            # TensorRT 8.x and newer
            serialized_engine = builder.build_serialized_network(network, config)
            if not serialized_engine:
                raise RuntimeError("Failed to build TensorRT engine")
            
            # Deserialize to get engine object
            runtime = trt.Runtime(self.trt_logger)
            engine = runtime.deserialize_cuda_engine(serialized_engine)
            
        except AttributeError:
            # TensorRT 7.x and older
            engine = builder.build_engine(network, config)
            if not engine:
                raise RuntimeError("Failed to build TensorRT engine")
            
            # Serialize to save
            serialized_engine = engine.serialize()
        
        build_time = time.time() - start_time
        self.logger.info(f"Engine built successfully in {build_time:.2f}s")
        
        # Save engine
        self.logger.info(f"Saving engine: {engine_path}")
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        
        # Engine info
        engine_size = os.path.getsize(engine_path) / (1024 * 1024)  # MB
        self.logger.info(f"Engine size: {engine_size:.2f} MB")
    
    def _create_calibrator(self, calibration_data):
        """Tạo INT8 calibrator"""
        return SimpleCalibrator(calibration_data, self.logger)
    
    def benchmark_engine(self, engine_path, input_shape=(1, 3, 640, 640), 
                        num_runs=100, warmup_runs=10):
        """Benchmark TensorRT engine performance"""
        
        if not os.path.exists(engine_path):
            self.logger.error(f"Engine không tồn tại: {engine_path}")
            return None
        
        self.logger.info(f"Benchmarking engine: {engine_path}")
        
        try:
            # Load engine
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            
            runtime = trt.Runtime(self.trt_logger)
            engine = runtime.deserialize_cuda_engine(engine_data)
            context = engine.create_execution_context()
            
            # Allocate buffers
            inputs, outputs, bindings, stream = self._allocate_buffers(engine)
            
            # Random input data
            input_data = np.random.random(input_shape).astype(np.float32)
            np.copyto(inputs[0].host, input_data.ravel())
            
            # Warmup
            self.logger.info(f"Warmup {warmup_runs} runs...")
            for _ in range(warmup_runs):
                self._do_inference(context, bindings, inputs, outputs, stream)
            
            # Benchmark
            self.logger.info(f"Benchmarking {num_runs} runs...")
            start_time = time.time()
            
            for _ in range(num_runs):
                self._do_inference(context, bindings, inputs, outputs, stream)
            
            total_time = time.time() - start_time
            avg_time = (total_time / num_runs) * 1000  # ms
            fps = 1000 / avg_time
            
            results = {
                'avg_inference_time_ms': avg_time,
                'fps': fps,
                'total_time_s': total_time,
                'num_runs': num_runs
            }
            
            self.logger.info(f"Benchmark results:")
            self.logger.info(f"   Average inference: {avg_time:.2f} ms")
            self.logger.info(f"   FPS: {fps:.1f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Benchmark error: {str(e)}")
            return None
    
    def _allocate_buffers(self, engine):
        """Allocate GPU buffers cho inference"""
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            
            # Handle different TensorRT versions for tensor info
            try:
                # TensorRT 8.x and newer
                tensor_shape = engine.get_tensor_shape(tensor_name)
                tensor_dtype = engine.get_tensor_dtype(tensor_name)
                tensor_mode = engine.get_tensor_mode(tensor_name)
                is_input = (tensor_mode == trt.TensorIOMode.INPUT)
            except AttributeError:
                # TensorRT 7.x and older fallback
                binding_idx = engine.get_binding_index(tensor_name)
                tensor_shape = engine.get_binding_shape(binding_idx)
                tensor_dtype = engine.get_binding_dtype(binding_idx)
                is_input = engine.binding_is_input(binding_idx)
            
            size = trt.volume(tensor_shape)
            dtype = trt.nptype(tensor_dtype)
            
            # Allocate host và device memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))
            
            if is_input:
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream
    
    def _do_inference(self, context, bindings, inputs, outputs, stream):
        """Run inference với TensorRT"""
        # Transfer input data to device
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        
        # Run inference
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        
        # Transfer predictions back
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        
        # Synchronize
        stream.synchronize()
        
        return [out.host for out in outputs]

class HostDeviceMem:
    """Helper class cho GPU memory management"""
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

if TRT_AVAILABLE:
    class SimpleCalibrator(trt.IInt8EntropyCalibrator2):
        """Simple INT8 calibrator"""
        
        def __init__(self, calibration_data, logger):
            trt.IInt8EntropyCalibrator2.__init__(self)
            self.cache_file = 'calibration.cache'
            self.data = calibration_data
            self.logger = logger
            self.batch_size = 1
            self.current_index = 0
            
            # Allocate device memory
            self.device_input = cuda.mem_alloc(1 * 3 * 640 * 640 * 4)  # FP32
        
        def get_batch_size(self):
            return self.batch_size
        
        def get_batch(self, names):
            if self.current_index >= len(self.data):
                return None
            
            # Get batch data
            batch_data = self.data[self.current_index:self.current_index + self.batch_size]
            self.current_index += self.batch_size
            
            # Copy to device
            cuda.memcpy_htod(self.device_input, batch_data[0])
            return [self.device_input]
        
        def read_calibration_cache(self):
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "rb") as f:
                    return f.read()
            return None
        
        def write_calibration_cache(self, cache):
            with open(self.cache_file, "wb") as f:
                f.write(cache)
else:
    class SimpleCalibrator:
        """Dummy calibrator when TensorRT not available"""
        def __init__(self, calibration_data, logger):
            pass

def create_calibration_data(video_path, num_frames=100):
    """Tạo calibration data từ video"""
    try:
        import cv2
    except ImportError:
        print("❌ OpenCV không khả dụng cho calibration data")
        return []
    
    cap = cv2.VideoCapture(video_path)
    calibration_data = []
    
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess giống YOLO
        frame = cv2.resize(frame, (640, 640))
        frame = frame.astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))  # HWC -> CHW
        frame = np.expand_dims(frame, axis=0)   # Add batch dim
        
        calibration_data.append(frame)
    
    cap.release()
    return calibration_data

# CLI utilities
def quantize_yolo_cli():
    """CLI để quantize YOLO models"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO TensorRT Quantization Tool')
    parser.add_argument('--model', required=True, help='Path to YOLO model (.pt)')
    parser.add_argument('--output', required=True, help='Output engine path (.engine)')
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'int8'], default='fp16')
    parser.add_argument('--calibration-video', help='Video for INT8 calibration')
    parser.add_argument('--workspace', type=int, default=1024, help='Workspace size (MB)')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark after quantization')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    quantizer = TensorRTQuantizer(logger)
    
    # Calibration data cho INT8
    calibration_data = None
    if args.precision == 'int8' and args.calibration_video:
        logger.info(f"📹 Creating calibration data from: {args.calibration_video}")
        calibration_data = create_calibration_data(args.calibration_video)
    
    # Quantize
    success = quantizer.quantize_model(
        model_path=args.model,
        engine_path=args.output,
        precision=args.precision,
        calibration_data=calibration_data,
        workspace_size=args.workspace * 1024 * 1024  # Convert MB to bytes
    )
    
    if success:
        logger.info(f"✅ Quantization hoàn thành: {args.output}")
        
        # Benchmark if requested
        if args.benchmark:
            results = quantizer.benchmark_engine(args.output)
            if results:
                logger.info(f"🏆 Performance: {results['fps']:.1f} FPS")
    else:
        logger.error("❌ Quantization thất bại")
        sys.exit(1)

if __name__ == "__main__":
    quantize_yolo_cli()
