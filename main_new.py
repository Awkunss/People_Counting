#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
People Counting System - Main Entry Point
Hệ thống đếm người với AI và web interface
"""

import sys
import os
import argparse

# Add app directory to path
current_dir = os.path.dirname(__file__)
app_dir = os.path.join(current_dir, 'app')
sys.path.insert(0, app_dir)

def show_methods():
    """Hiển thị các phương pháp có sẵn"""
    print("=" * 50)
    print("🔍 PEOPLE COUNTING SYSTEM")
    print("=" * 50)
    print("📊 LINE: Đếm người qua đường thẳng")
    print("🏢 ZONE: Đếm người trong vùng khu vực")
    print("🌐 WEB:  Giao diện web realtime")
    print()
    print("💡 Usage:")
    print("  python main.py --method line --video data/Test.mp4")
    print("  python main.py --method zone --video 0")
    print("  python main.py --method web")
    print("=" * 50)

def run_line_counting(video, model):
    """Chạy line counting"""
    try:
        from core.line_counting import line_counter
        print("🎯 Starting LINE counting...")
        line_counter(video, model)
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

def run_zone_counting(video, model):
    """Chạy zone counting"""
    try:
        from core.zone_counting import zone_counter
        print("🏢 Starting ZONE counting...")
        zone_counter(video, model)
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

def run_web_server():
    """Chạy web server"""
    try:
        from web.server import app, socketio
        from config.settings import WEB_HOST, WEB_PORT, WEB_DEBUG
        
        print("🌐 Starting Web Server...")
        print("🚀 Launching web server...")
        print(f"🌍 URL: http://localhost:{WEB_PORT}")
        print(f"📱 Mobile: http://<your-ip>:{WEB_PORT}")
        print("⚠️  Nhấn Ctrl+C để dừng server")
        
        # Start Flask-SocketIO server
        socketio.run(
            app, 
            host=WEB_HOST, 
            port=WEB_PORT, 
            debug=WEB_DEBUG,
            allow_unsafe_werkzeug=True
        )
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("📝 Hãy cài đặt: pip install flask flask-socketio")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='People Counting System')
    parser.add_argument('--method', choices=['line', 'zone', 'web'], 
                       help='Counting method')
    parser.add_argument('--video', default='Test.mp4',
                       help='Video source (file path or camera ID)')
    parser.add_argument('--model', default='yolov8s.pt',
                       help='YOLO model to use')
    parser.add_argument('--show-methods', action='store_true',
                       help='Show available methods')
    
    args = parser.parse_args()
    
    if args.show_methods or not args.method:
        show_methods()
        return
    
    if args.method == 'line':
        run_line_counting(args.video, args.model)
    elif args.method == 'zone':
        run_zone_counting(args.video, args.model)
    elif args.method == 'web':
        run_web_server()

if __name__ == "__main__":
    main()
