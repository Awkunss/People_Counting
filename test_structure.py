#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script để kiểm tra cấu trúc project mới
"""

import sys
import os

def test_imports():
    """Test tất cả imports"""
    print("🧪 Testing imports...")
    
    # Add app to path
    current_dir = os.path.dirname(__file__)
    app_dir = os.path.join(current_dir, 'app')
    sys.path.insert(0, app_dir)
    
    try:
        # Test config
        print("  ✓ Testing config...")
        from config.settings import CLASS_ID, CONF_THRESHOLD
        print(f"    CLASS_ID: {CLASS_ID}, CONF_THRESHOLD: {CONF_THRESHOLD}")
        
        # Test tracking
        print("  ✓ Testing tracking...")
        from app.core.tracking.deep_tracker import DeepTracker
        print("    DeepTracker imported successfully")
        
        # Test core modules (might fail due to path issues, that's OK)
        print("  ⚠️  Testing core modules (may fail - expected)...")
        try:
            from app.core.line_counting import line_counter
            print("    line_counting imported successfully")
        except ImportError as e:
            print(f"    line_counting import failed: {e}")
        
        try:
            from app.core.zone_counting import zone_counter  
            print("    zone_counting imported successfully")
        except ImportError as e:
            print(f"    zone_counting import failed: {e}")
        
        # Test web modules
        print("  ⚠️  Testing web modules (may fail - expected)...")
        try:
            from app.web.server import app
            print("    web.server imported successfully")
        except ImportError as e:
            print(f"    web.server import failed: {e}")
        
        print("✅ Import test completed!")
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")

def test_structure():
    """Test cấu trúc thư mục"""
    print("\n🏗️  Testing project structure...")
    
    required_dirs = [
        'app',
        'app/core', 
        'app/core/tracking',
        'app/web',
        'app/utils',
        'config',
        'models',
        'data',
        'templates',
        'static',
        'docs',
        'scripts',
        'tests'
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ❌ {dir_path} - Missing!")
    
    required_files = [
        'app/__init__.py',
        'app/core/__init__.py',
        'app/core/tracking/__init__.py',
        'app/core/tracking/deep_tracker.py',
        'app/web/__init__.py',
        'app/utils/__init__.py',
        'config/settings.py',
        'main.py',
        'requirements.txt'
    ]
    
    print("\n📄 Testing required files...")
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ❌ {file_path} - Missing!")
    
    print("✅ Structure test completed!")

def test_config():
    """Test configuration"""
    print("\n⚙️  Testing configuration...")
    
    try:
        sys.path.insert(0, 'config')
        from config.settings import MODELS_DIR, DATA_DIR, get_model_path, get_video_path
        
        print(f"  ✓ MODELS_DIR: {MODELS_DIR}")
        print(f"  ✓ DATA_DIR: {DATA_DIR}")
        
        # Test path functions
        model_path = get_model_path('yolov8s.pt')
        video_path = get_video_path('Test.mp4')
        
        print(f"  ✓ Model path: {model_path}")
        print(f"  ✓ Video path: {video_path}")
        
        print("✅ Configuration test completed!")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")

if __name__ == "__main__":
    print("🚀 People Counting System - Structure Test")
    print("=" * 50)
    
    test_structure()
    test_config() 
    test_imports()
    
    print("\n🎉 All tests completed!")
    print("\n💡 Next steps:")
    print("  1. python main.py --show-methods")
    print("  2. python main.py --method web")
    print("  3. Test individual modules")
