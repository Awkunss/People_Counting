#!/usr/bin/env python3
"""
Cleanup script cho PyCUDA context và deprecation warnings
"""

import warnings
import sys
import os

def suppress_warnings():
    """Suppress common warnings"""
    # Suppress torchvision deprecation warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
    
    # Suppress PyCUDA warnings
    warnings.filterwarnings("ignore", message=".*CUDA installation directory.*")
    
    print("✅ Warnings suppressed")

def setup_cuda_cleanup():
    """Setup proper CUDA context cleanup"""
    import atexit
    
    def cleanup_cuda():
        try:
            import pycuda.driver as cuda
            # Get current context and pop it
            ctx = cuda.Context.get_current()
            if ctx:
                ctx.pop()
                print("✅ CUDA context cleaned up")
        except:
            pass
    
    atexit.register(cleanup_cuda)

if __name__ == "__main__":
    suppress_warnings()
    setup_cuda_cleanup()
    print("🧹 Cleanup script ready")
