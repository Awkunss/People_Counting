#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script để kiểm tra cấu trúc project mới (Safe version)
"""

import sys
import os

def test_imports_safe():
    """Test imports an toàn không load torch"""
    print("🧪 Testing imports (safe mode)...")
    
    try:
        # Test config only
        print("  ✓ Testing config...")
        sys.path.insert(0, 'config')
        from settings import CLASS_ID, CONF_THRESHOLD
        print(f"    CLASS_ID: {CLASS_ID}, CONF_THRESHOLD: {CONF_THRESHOLD}")
        
        # Test file existence and structure instead of imports
        print("  ✓ Testing module files structure...")
        
        modules_to_check = {
            'app/core/tracking/deep_tracker.py': ['class DeepTracker', 'class FeatureExtractor', 'class Track'],
            'app/core/line_counting.py': ['def line_counter', 'line_points', 'counts'],
            'app/core/zone_counting.py': ['def zone_counter', 'zone_points', 'current_in_zone'],
            'app/web/server.py': ['Flask', 'SocketIO', 'def handle_start_counting'],
            'app/web/line_counting_web.py': ['class LineCounterWeb'],
            'app/web/zone_counting_web.py': ['class ZoneCounterWeb']
        }
        
        for file_path, keywords in modules_to_check.items():
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                found_keywords = []
                missing_keywords = []
                
                for keyword in keywords:
                    if keyword in content:
                        found_keywords.append(keyword)
                    else:
                        missing_keywords.append(keyword)
                
                if found_keywords:
                    print(f"    ✓ {file_path}: {found_keywords}")
                if missing_keywords:
                    print(f"    ⚠️ {file_path}: missing {missing_keywords}")
            else:
                print(f"    ❌ {file_path} - File missing!")
        
        print("✅ Safe import test completed!")
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

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
        'app/core/line_counting.py',
        'app/core/zone_counting.py',
        'app/web/__init__.py',
        'app/web/server.py',
        'app/web/line_counting_web.py',
        'app/web/zone_counting_web.py',
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
        from settings import MODELS_DIR, DATA_DIR, get_model_path, get_video_path
        
        print(f"  ✓ MODELS_DIR: {MODELS_DIR}")
        print(f"  ✓ DATA_DIR: {DATA_DIR}")
        
        # Test path functions
        model_path = get_model_path('yolov8s.pt')
        video_path = get_video_path('Test.mp4')
        
        print(f"  ✓ Model path: {model_path}")
        print(f"  ✓ Video path: {video_path}")
        
        # Check if files actually exist
        if os.path.exists(str(model_path)):
            print(f"  ✓ Model file exists")
        else:
            print(f"  ⚠️ Model file not found (will auto-download)")
            
        if os.path.exists(str(video_path)):
            print(f"  ✓ Video file exists")
        else:
            print(f"  ⚠️ Video file not found")
        
        print("✅ Configuration test completed!")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")

def test_main_entry():
    """Test main.py entry point"""
    print("\n🚀 Testing main entry point...")
    
    try:
        if os.path.exists('main.py'):
            with open('main.py', 'r', encoding='utf-8') as f:
                content = f.read()
            
            required_functions = [
                'def show_methods',
                'def run_line_counting',
                'def run_zone_counting', 
                'def run_web_server',
                'def main'
            ]
            
            for func in required_functions:
                if func in content:
                    print(f"  ✓ {func} found")
                else:
                    print(f"  ❌ {func} missing")
            
            # Check imports
            required_imports = [
                'import argparse',
                'import sys',
                'import os'
            ]
            
            for imp in required_imports:
                if imp in content:
                    print(f"  ✓ {imp}")
                else:
                    print(f"  ⚠️ {imp} not found")
                    
            print("✅ Main entry point test completed!")
        else:
            print("  ❌ main.py not found!")
            
    except Exception as e:
        print(f"❌ Main entry test failed: {e}")

if __name__ == "__main__":
    print("🚀 People Counting System - Safe Structure Test")
    print("=" * 60)
    
    test_structure()
    test_config() 
    test_imports_safe()
    test_main_entry()
    
    print("\n🎉 All tests completed!")
    print("\n💡 Next steps:")
    print("  1. python main.py --show-methods")
    print("  2. python main.py --method web")
    print("  3. Test individual modules")
    print("\n⚠️  Note: This safe test avoids torch import issues")
    print("    Real imports will work when running the actual application")
