#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mock TensorRT Module for Testing
Module giả lập TensorRT để test system khi chưa cài TensorRT thật
"""

import numpy as np
import logging

# Mock TensorRT classes và functions
class MockTensorRT:
    """Mock TensorRT module"""
    __version__ = "8.6.1-mock"
    
    class Logger:
        WARNING = 1
        def __init__(self, level):
            self.level = level
    
    class Builder:
        def __init__(self, logger):
            self.logger = logger
            self.platform_has_fast_fp16 = True
            self.platform_has_fast_int8 = True
        
        def create_builder_config(self):
            return MockBuilderConfig()
        
        def create_network(self, flags):
            return MockNetwork()
        
        def build_engine(self, network, config):
            return MockEngine()
    
    class OnnxParser:
        def __init__(self, network, logger):
            self.network = network
            self.logger = logger
            self.num_errors = 0
        
        def parse(self, data):
            return True
        
        def get_error(self, i):
            return "Mock error"
    
    class NetworkDefinitionCreationFlag:
        EXPLICIT_BATCH = 1
    
    class BuilderFlag:
        FP16 = 1
        INT8 = 2
    
    class Runtime:
        def __init__(self, logger):
            self.logger = logger
        
        def deserialize_cuda_engine(self, data):
            return MockEngine()

class MockBuilderConfig:
    def __init__(self):
        self.max_workspace_size = 1024 * 1024 * 1024
    
    def set_flag(self, flag):
        pass
    
    def add_optimization_profile(self, profile):
        pass

class MockNetwork:
    def __init__(self):
        self.inputs = [MockTensor()]
    
    def get_input(self, index):
        return self.inputs[index]

class MockTensor:
    def __init__(self):
        self.name = "input"
        self.shape = [1, 3, 640, 640]

class MockEngine:
    def __init__(self):
        self.max_batch_size = 1
        self.bindings = ["input", "output"]
    
    def __iter__(self):
        return iter(self.bindings)
    
    def get_binding_shape(self, binding):
        return [1, 3, 640, 640]
    
    def get_binding_dtype(self, binding):
        return np.float32
    
    def binding_is_input(self, binding):
        return binding == "input"
    
    def serialize(self):
        return b"mock_engine_data"
    
    def create_execution_context(self):
        return MockContext()

class MockContext:
    def execute_async_v2(self, bindings, stream_handle):
        return True

class MockOptimizationProfile:
    def set_shape(self, name, min_shape, opt_shape, max_shape):
        pass

class MockPyCUDA:
    """Mock PyCUDA module"""
    
    class driver:
        @staticmethod
        def Device(id):
            return MockDevice()
        
        class Stream:
            def __init__(self):
                self.handle = 0
            
            def synchronize(self):
                pass
        
        @staticmethod
        def pagelocked_empty(size, dtype):
            return np.zeros(size, dtype=dtype)
        
        @staticmethod
        def mem_alloc(size):
            return MockDeviceMemory(size)
        
        @staticmethod
        def memcpy_htod_async(dest, src, stream):
            pass
        
        @staticmethod
        def memcpy_dtoh_async(dest, src, stream):
            pass
        
        @staticmethod
        def memcpy_htod(dest, src):
            pass

class MockDevice:
    def name(self):
        return "Mock GPU Device"

class MockDeviceMemory:
    def __init__(self, size):
        self.size = size
        self.ptr = id(self)  # Fake pointer
    
    def __int__(self):
        return self.ptr

def install_mock_tensorrt():
    """Install mock TensorRT modules"""
    import sys
    
    # Mock tensorrt
    sys.modules['tensorrt'] = MockTensorRT()
    
    # Mock pycuda
    mock_pycuda = type('MockPyCUDA', (), {})()
    mock_pycuda.driver = MockPyCUDA.driver
    mock_pycuda.autoinit = type('MockAutoInit', (), {})()
    
    sys.modules['pycuda'] = mock_pycuda
    sys.modules['pycuda.driver'] = MockPyCUDA.driver
    sys.modules['pycuda.autoinit'] = mock_pycuda.autoinit
    
    print("✅ Mock TensorRT installed")
    print("⚠️ This is for testing only - not real TensorRT acceleration")

if __name__ == "__main__":
    install_mock_tensorrt()
    
    # Test imports
    import tensorrt as trt
    import pycuda.driver as cuda
    
    print(f"TensorRT version: {trt.__version__}")
    print("PyCUDA driver imported successfully")
    print("🎉 Mock TensorRT test successful!")
