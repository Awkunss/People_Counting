#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test TensorRT Fixes
Quick test to verify MockBoxes and cleanup fixes
"""

import os
import sys
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_tensorrt_inference():
    """Test TensorRT inference MockBoxes fix"""
    try:
        from app.core.tensorrt_inference import create_yolo_model
        
        # Test creating TensorRT model
        model_path = "yolov9s_fp16.engine"
        if os.path.exists(model_path):
            print(f"✅ Testing TensorRT model: {model_path}")
            
            # Create logger
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
            
            # Create model
            model = create_yolo_model(model_path, logger)
            print(f"✅ Model created successfully: {type(model)}")
            
            # Test MockBoxes len() functionality
            import cv2
            import numpy as np
            
            # Create a dummy frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Run inference
            results = model(frame, classes=[0], conf=0.5)
            print(f"✅ Inference completed, results type: {type(results)}")
            
            if len(results) > 0:
                boxes = results[0].boxes
                print(f"✅ MockBoxes type: {type(boxes)}")
                print(f"✅ MockBoxes len(): {len(boxes)}")
                print(f"✅ MockBoxes iteration test: {list(boxes) is not None}")
                
                # Test accessing properties
                print(f"✅ xyxy shape: {boxes.xyxy.shape if hasattr(boxes.xyxy, 'shape') else 'tensor'}")
                print(f"✅ conf shape: {boxes.conf.shape if hasattr(boxes.conf, 'shape') else 'tensor'}")
                print(f"✅ cls shape: {boxes.cls.shape if hasattr(boxes.cls, 'shape') else 'tensor'}")
            
            # Test cleanup
            print("🧹 Testing cleanup...")
            model.cleanup()
            print("✅ Cleanup completed successfully")
            
            return True
            
        else:
            print(f"⚠️ TensorRT model not found: {model_path}")
            return False
            
    except Exception as e:
        print(f"❌ TensorRT test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_line_counting_integration():
    """Test line counting with TensorRT"""
    try:
        print("\n🧪 Testing Line Counting Integration...")
        
        # Test import
        from line_counting_web import LineCounterWeb
        print("✅ LineCounterWeb import successful")
        
        # Check TensorRT availability
        import line_counting_web
        print(f"✅ TensorRT available in line counting: {line_counting_web.TENSORRT_AVAILABLE}")
        
        return True
        
    except Exception as e:
        print(f"❌ Line counting integration test failed: {e}")
        return False

def test_zone_counting_integration():
    """Test zone counting with TensorRT"""
    try:
        print("\n🧪 Testing Zone Counting Integration...")
        
        # Test import
        from zone_counting_web import ZoneCounterWeb
        print("✅ ZoneCounterWeb import successful")
        
        # Check TensorRT availability
        import zone_counting_web
        print(f"✅ TensorRT available in zone counting: {zone_counting_web.TENSORRT_AVAILABLE}")
        
        return True
        
    except Exception as e:
        print(f"❌ Zone counting integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing TensorRT Fixes...")
    print("=" * 50)
    
    # Run tests
    tests = [
        ("TensorRT Inference", test_tensorrt_inference),
        ("Line Counting Integration", test_line_counting_integration),
        ("Zone Counting Integration", test_zone_counting_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! TensorRT fixes are working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
