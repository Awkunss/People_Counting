#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
People Counting System - Main Dispatcher
Hệ thống đếm người với hai phương pháp: Line và Zone
"""

import argparse
import sys

# Import các module counting
try:
    from line_counting import line_counter
    from zone_counting import zone_counter
except ImportError as e:
    print(f"❌ Lỗi import: {e}")
    sys.exit(1)

class PeopleCountingDispatcher:
    """Lớp điều phối đơn giản cho hệ thống đếm người"""
    
    def __init__(self):
        self.methods = ['line', 'zone']
    
    def show_methods_info(self):
        """Hiển thị thông tin về các phương pháp counting"""
        print("\n" + "="*50)
        print("🔍 PEOPLE COUNTING SYSTEM")
        print("="*50)
        print("📊 LINE: Đếm người qua đường thẳng")
        print("🏢 ZONE: Đếm người trong vùng khu vực")
        print("\n💡 Usage:")
        print("  python main.py --method line --video Test.mp4")
        print("  python main.py --method zone --video 0")
        print("="*50)
    
    def dispatch(self, method: str, video_path: str, model_path: str) -> None:
        """Điều phối chạy phương pháp counting"""
        print(f"\n🚀 Starting {method.upper()} counting...")
        
        # Chuyển đổi camera ID nếu cần
        if video_path.isdigit():
            video_path = int(video_path)
        
        # Chạy phương pháp tương ứng
        if method == 'line':
            line_counter(video_path=video_path, model=model_path)
        elif method == 'zone':
            zone_counter(video_path=video_path, model=model_path)
        else:
            print(f"❌ Unknown method: {method}")
            print(f"✅ Available: {self.methods}")

def create_argument_parser() -> argparse.ArgumentParser:
    """Tạo argument parser đơn giản"""
    parser = argparse.ArgumentParser(description="🔍 People Counting System")
    
    parser.add_argument('--method', '-m', 
                       choices=['line', 'zone'], 
                       default='line',
                       help='Phương pháp: line hoặc zone')
    
    parser.add_argument('--video', '-v', 
                       type=str, 
                       default='Test.mp4',
                       help='Video file hoặc camera ID (0, 1, 2...)')
    
    parser.add_argument('--model', 
                       type=str, 
                       default='yolov8s.pt',
                       help='YOLO model file')
    
    parser.add_argument('--show-methods', 
                       action='store_true',
                       help='Hiển thị thông tin methods')
    
    return parser

def main():
    """Hàm main đơn giản"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    dispatcher = PeopleCountingDispatcher()
    
    # Hiển thị methods info nếu được yêu cầu
    if args.show_methods:
        dispatcher.show_methods_info()
        return
    
    # Chạy dispatcher
    try:
        dispatcher.dispatch(args.method, args.video, args.model)
    except KeyboardInterrupt:
        print("\n⚠️  Dừng bởi người dùng")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")

if __name__ == "__main__":
    main()