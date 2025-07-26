#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorRT Inference Wrapper
Wrapper để chạy inference với TensorRT engines
"""

import numpy as np
import cv2
import time
import logging
from pathlib import Path

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

try:
    from ultralytics import YOLO
    from ultralytics.engine.results import Results
    import torch
except ImportError as e:
    logging.error(f"Missing YOLO dependencies: {e}")

class TensorRTInference:
    """TensorRT inference wrapper tương thích với YOLO interface"""
    
    def __init__(self, engine_path, logger=None):
        self.engine_path = engine_path
        self.logger = logger or logging.getLogger(__name__)
        
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT không khả dụng")
        
        # Initialize CUDA context first
        self._init_cuda_context()
        
        # Load engine
        self.engine = None
        self.context = None
        self.inputs = None
        self.outputs = None
        self.bindings = None
        self.stream = None
        
        self._load_engine()
        self._allocate_buffers()
        
        # Store context for cleanup
        self._cuda_context = None
        try:
            import pycuda.driver as cuda
            try:
                self._cuda_context = cuda.Context.get_current()
            except:
                pass
        except:
            pass
        
        self.logger.info(f"✅ TensorRT engine loaded: {engine_path}")
    
    def _init_cuda_context(self):
        """Initialize CUDA context properly"""
        try:
            # Import here to avoid issues if not available
            import pycuda.driver as cuda
            
            # Initialize CUDA driver
            cuda.init()
            
            # Get device and create context if none exists
            device_count = cuda.Device.count()
            if device_count == 0:
                raise RuntimeError("No CUDA devices found")
            
            device = cuda.Device(0)  # Use first GPU
            
            # Check if context already exists
            try:
                current_ctx = cuda.Context.get_current()
                if current_ctx:
                    self.logger.info("✅ Using existing CUDA context")
                    return
            except:
                pass
            
            # Create new context
            ctx = device.make_context()
            self.logger.info("✅ CUDA context created successfully")
            
        except Exception as e:
            self.logger.warning(f"⚠️ CUDA context init warning: {e}")
            # Try pycuda.autoinit as fallback
            try:
                import pycuda.autoinit
                self.logger.info("✅ Used pycuda.autoinit fallback")
            except:
                self.logger.error("❌ Failed to initialize CUDA context")
    
    def _load_engine(self):
        """Load TensorRT engine"""
        trt_logger = trt.Logger(trt.Logger.WARNING)
        
        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(trt_logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        if not self.engine or not self.context:
            raise RuntimeError(f"Failed to load TensorRT engine: {self.engine_path}")
    
    def _allocate_buffers(self):
        """Allocate GPU buffers"""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        # Handle different TensorRT versions
        try:
            # TensorRT 8.x and newer
            for i in range(self.engine.num_io_tensors):
                tensor_name = self.engine.get_tensor_name(i)
                tensor_shape = self.engine.get_tensor_shape(tensor_name)
                tensor_dtype = self.engine.get_tensor_dtype(tensor_name)
                tensor_mode = self.engine.get_tensor_mode(tensor_name)
                is_input = (tensor_mode == trt.TensorIOMode.INPUT)
                
                size = trt.volume(tensor_shape)
                dtype = trt.nptype(tensor_dtype)
                
                # Allocate host và device memory
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                
                self.bindings.append(int(device_mem))
                
                if is_input:
                    self.inputs.append(HostDeviceMem(host_mem, device_mem))
                else:
                    self.outputs.append(HostDeviceMem(host_mem, device_mem))
                    
        except AttributeError:
            # TensorRT 7.x and older fallback
            for binding in self.engine:
                binding_idx = self.engine.get_binding_index(binding)
                size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
                dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                
                # Allocate host và device memory
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                
                self.bindings.append(int(device_mem))
                
                if self.engine.binding_is_input(binding):
                    self.inputs.append(HostDeviceMem(host_mem, device_mem))
                else:
                    self.outputs.append(HostDeviceMem(host_mem, device_mem))
    
    def __call__(self, source, classes=None, conf=0.25, verbose=False, **kwargs):
        """
        Inference method tương thích với YOLO interface
        
        Args:
            source: Input image hoặc batch images
            classes: List of class IDs to filter
            conf: Confidence threshold
            verbose: Verbose logging
        
        Returns:
            List of Results objects tương thích với YOLO
        """
        if isinstance(source, (str, Path)):
            # Load image from path
            image = cv2.imread(str(source))
            images = [image]
        elif isinstance(source, np.ndarray):
            if len(source.shape) == 3:
                images = [source]
            else:
                images = [source]  # Batch
        else:
            images = [source]
        
        results = []
        
        for image in images:
            # Preprocess
            input_tensor = self._preprocess(image)
            
            # Inference
            outputs = self._inference(input_tensor)
            
            # Postprocess
            detections = self._postprocess(outputs, image.shape, conf, classes)
            
            # Create Results object
            result = self._create_results(detections, image)
            results.append(result)
        
        return results
    
    def _preprocess(self, image):
        """Preprocess image cho TensorRT input"""
        # Resize về 640x640
        input_size = (640, 640)
        resized = cv2.resize(image, input_size)
        
        # Normalize
        input_tensor = resized.astype(np.float32) / 255.0
        
        # HWC -> CHW
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        
        # Add batch dimension
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor
    
    def _inference(self, input_tensor):
        """Run TensorRT inference"""
        # Copy input to device
        np.copyto(self.inputs[0].host, input_tensor.ravel())
        
        # Transfer input data to device
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
        
        # Handle different TensorRT versions for execution
        try:
            # TensorRT 8.x and newer - set tensor addresses
            for i, inp in enumerate(self.inputs):
                tensor_name = self.engine.get_tensor_name(i)
                self.context.set_tensor_address(tensor_name, inp.device)
            
            for i, out in enumerate(self.outputs):
                tensor_name = self.engine.get_tensor_name(len(self.inputs) + i)
                self.context.set_tensor_address(tensor_name, out.device)
            
            # Execute inference
            self.context.execute_async_v3(stream_handle=self.stream.handle)
            
        except AttributeError:
            try:
                # TensorRT 7.x fallback
                self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            except AttributeError:
                # Very old TensorRT versions
                self.context.execute_async(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Transfer predictions back
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
        
        # Synchronize
        self.stream.synchronize()
        
        # Get output
        output = self.outputs[0].host.copy()
        
        # Reshape output based on YOLO format
        # YOLOv9 output shape: [1, 84, 8400] hoặc tương tự
        if len(output.shape) == 1:
            # Flat output, cần reshape
            # Thông thường: [batch, num_classes + 4, num_detections]
            output = output.reshape(1, -1, 8400)  # Adjust based on actual model
        
        return output
    
    def _postprocess(self, outputs, orig_shape, conf_threshold, classes):
        """Postprocess TensorRT outputs thành detections"""
        detections = []
        
        # outputs shape: [1, 84, 8400] for YOLOv9
        # 84 = 4 (bbox) + 80 (classes)
        pred = outputs[0]  # Remove batch dimension
        
        # Transpose để dễ xử lý: [8400, 84]
        pred = pred.transpose()
        
        # Extract boxes và scores
        boxes = pred[:, :4]  # x, y, w, h
        scores = pred[:, 4:]  # class scores
        
        # Get max scores và class indices
        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)
        
        # Filter by confidence
        valid_indices = max_scores > conf_threshold
        
        if not np.any(valid_indices):
            return detections
        
        boxes = boxes[valid_indices]
        max_scores = max_scores[valid_indices]
        class_ids = class_ids[valid_indices]
        
        # Filter by classes if specified
        if classes is not None:
            class_filter = np.isin(class_ids, classes)
            boxes = boxes[class_filter]
            max_scores = max_scores[class_filter]
            class_ids = class_ids[class_filter]
        
        if len(boxes) == 0:
            return detections
        
        # Convert center format to corner format và scale
        h_orig, w_orig = orig_shape[:2]
        scale_x = w_orig / 640
        scale_y = h_orig / 640
        
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        
        x1 = (x_center - width / 2) * scale_x
        y1 = (y_center - height / 2) * scale_y
        x2 = (x_center + width / 2) * scale_x
        y2 = (y_center + height / 2) * scale_y
        
        # Apply NMS
        indices = self._nms(
            np.column_stack([x1, y1, x2, y2]),
            max_scores,
            iou_threshold=0.45
        )
        
        # Create final detections
        for i in indices:
            detections.append({
                'bbox': [x1[i], y1[i], x2[i], y2[i]],
                'confidence': max_scores[i],
                'class_id': class_ids[i]
            })
        
        return detections
    
    def _nms(self, boxes, scores, iou_threshold=0.45):
        """Non-Maximum Suppression"""
        if len(boxes) == 0:
            return []
        
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        indices = np.argsort(scores)[::-1]
        
        keep = []
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            indices = indices[1:]
            
            # Calculate IoU
            xx1 = np.maximum(x1[current], x1[indices])
            yy1 = np.maximum(y1[current], y1[indices])
            xx2 = np.minimum(x2[current], x2[indices])
            yy2 = np.minimum(y2[current], y2[indices])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h
            
            union = areas[current] + areas[indices] - intersection
            iou = intersection / (union + 1e-6)
            
            # Keep boxes with IoU less than threshold
            indices = indices[iou <= iou_threshold]
        
        return keep
    
    def _create_results(self, detections, orig_img):
        """Create Results object tương thích với YOLO"""
        if len(detections) == 0:
            # Empty results
            boxes = torch.empty((0, 4))
            scores = torch.empty((0,))
            class_ids = torch.empty((0,), dtype=torch.long)
        else:
            boxes = torch.tensor([det['bbox'] for det in detections])
            scores = torch.tensor([det['confidence'] for det in detections])
            class_ids = torch.tensor([det['class_id'] for det in detections], dtype=torch.long)
        
        # Create mock Results object
        class MockBoxes:
            def __init__(self, boxes, scores, class_ids):
                self.xyxy = boxes
                self.conf = scores
                self.cls = class_ids
                self._data = boxes  # Store for len() compatibility
            
            def __len__(self):
                """Return number of detections"""
                return len(self._data) if self._data is not None else 0
            
            def __iter__(self):
                """Make iterable"""
                return iter(self._data) if self._data is not None else iter([])
        
        class MockResults:
            def __init__(self, boxes):
                self.boxes = boxes
        
        mock_boxes = MockBoxes(boxes, scores, class_ids)
        return MockResults(mock_boxes)
    
    def cleanup(self):
        """Clean up CUDA resources"""
        try:
            if hasattr(self, 'stream') and self.stream:
                self.stream.synchronize()
            
            # Clean up device memory
            if hasattr(self, 'inputs'):
                for inp in self.inputs:
                    if hasattr(inp, 'device') and inp.device:
                        inp.device.free()
            
            if hasattr(self, 'outputs'):
                for out in self.outputs:
                    if hasattr(out, 'device') and out.device:
                        out.device.free()
            
            # Clean up context
            if hasattr(self, 'context') and self.context:
                del self.context
            
            if hasattr(self, 'engine') and self.engine:
                del self.engine
                
            self.logger.info("✅ TensorRT resources cleaned up")
            
        except Exception as e:
            self.logger.warning(f"⚠️ TensorRT cleanup warning: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass

class HostDeviceMem:
    """Simple helper để store host và device memory pointers"""
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

class YOLOWithTensorRT:
    """YOLO wrapper hỗ trợ cả PyTorch và TensorRT models"""
    
    def __init__(self, model_path, logger=None):
        self.model_path = model_path
        self.logger = logger or logging.getLogger(__name__)
        
        # Determine model type
        if model_path.endswith('.engine'):
            if not TRT_AVAILABLE:
                raise RuntimeError("TensorRT không khả dụng cho .engine model")
            self.model = TensorRTInference(model_path, logger)
            self.is_tensorrt = True
            self.logger.info(f"🚀 Loaded TensorRT model: {model_path}")
        else:
            self.model = YOLO(model_path)
            self.is_tensorrt = False
            self.logger.info(f"📦 Loaded PyTorch model: {model_path}")
    
    def __call__(self, *args, **kwargs):
        """Forward call to appropriate model"""
        return self.model(*args, **kwargs)
    
    def predict(self, *args, **kwargs):
        """Predict method"""
        if hasattr(self.model, 'predict'):
            return self.model.predict(*args, **kwargs)
        else:
            return self.model(*args, **kwargs)
    
    def cleanup(self):
        """Clean up resources"""
        if self.is_tensorrt and hasattr(self.model, 'cleanup'):
            self.model.cleanup()
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass

# Factory function
def create_yolo_model(model_path, logger=None):
    """Factory function để tạo YOLO model with TensorRT support"""
    return YOLOWithTensorRT(model_path, logger)
