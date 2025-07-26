#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
People Counting Web Server - Flask + Socket.IO
Hệ thống đếm người với giao diện web realtime
"""

import cv2
import base64
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import threading
import time
import queue
import sys
import os
from datetime import datetime
import traceback

# Add paths for imports
current_dir = os.path.dirname(__file__)
app_dir = os.path.dirname(current_dir)
config_dir = os.path.join(app_dir, '..', 'config')
templates_dir = os.path.join(app_dir, '..', 'templates')

sys.path.insert(0, config_dir)
sys.path.insert(0, current_dir)

# Import configuration
from config.settings import WEB_HOST, WEB_PORT, WEB_DEBUG, get_model_path, get_video_path

# Import web counting modules
try:
    from line_counting_web import LineCounterWeb
    from zone_counting_web import ZoneCounterWeb
except ImportError as e:
    print(f"❌ Lỗi import web modules: {e}")
    print("📝 Tạo các web modules...")

app = Flask(__name__, template_folder=templates_dir)
app.config['SECRET_KEY'] = 'people_counting_secret_key_2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables
current_counter = None
is_running = False
stats_queue = queue.Queue()
log_queue = queue.Queue()

class WebLogger:
    """Logger để gửi log realtime qua Socket.IO"""
    
    def __init__(self, socketio_instance):
        self.socketio = socketio_instance
    
    def log(self, level, message):
        """Gửi log message"""
        try:
            self.socketio.emit('log_message', {
                'level': level,
                'message': message,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
        except Exception as e:
            print(f"Logging error: {e}")
    
    def info(self, message):
        self.log('info', f"ℹ️ {message}")
    
    def warning(self, message):
        self.log('warning', f"⚠️ {message}")
    
    def error(self, message):
        self.log('error', f"❌ {message}")
    
    def success(self, message):
        self.log('info', f"✅ {message}")

logger = WebLogger(socketio)

@app.route('/')
def index():
    """Trang chính"""
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    """Xử lý kết nối client"""
    logger.success('Client đã kết nối')
    emit('system_status', {'status': 'stopped', 'message': 'Đã dừng'})

@socketio.on('disconnect')
def handle_disconnect():
    """Xử lý ngắt kết nối client"""
    logger.warning('Client đã ngắt kết nối')

@socketio.on('start_counting')
def handle_start_counting(data):
    """Bắt đầu counting"""
    global current_counter, is_running
    
    try:
        if is_running:
            logger.warning('Hệ thống đang chạy!')
            return
        
        method = data.get('method', 'line')
        video_path = data.get('video', 'Test.mp4')
        model_path = data.get('model', 'yolov8s.pt')
        
        # Chuyển đổi camera ID nếu cần
        if video_path.isdigit():
            video_path = int(video_path)
        
        print(f"🚀 Starting counting - Method: {method}, Video: {video_path}, Model: {model_path}")  # Debug log
        logger.info(f'Khởi động {method} counting với {model_path}')
        
        # Tạo counter instance
        if method == 'line':
            current_counter = LineCounterWeb(
                video_path=video_path,
                model_path=model_path,
                socketio=socketio,
                logger=logger
            )
            # Bắt đầu counting trong thread riêng
            counting_thread = threading.Thread(target=run_counting)
            counting_thread.daemon = True
            counting_thread.start()
        elif method == 'zone':
            print("🏢 Creating ZoneCounterWeb instance...")  # Debug log
            # Sử dụng ZoneCounterWeb
            current_counter = ZoneCounterWeb(
                video_path=video_path,
                model_path=model_path,
                socketio=socketio,
                logger=logger
            )
            print("✅ ZoneCounterWeb created, starting thread...")  # Debug log
            # Bắt đầu counting trong thread riêng
            counting_thread = threading.Thread(target=run_counting)
            counting_thread.daemon = True
            counting_thread.start()
        else:
            logger.error(f'Phương pháp không hỗ trợ: {method}')
            return
        
        is_running = True
        emit('system_status', {'status': 'running', 'message': f'Đang chạy {method}'})
        logger.success(f'Đã bắt đầu {method} counting')
        
    except Exception as e:
        logger.error(f'Lỗi khởi động: {str(e)}')
        emit('system_status', {'status': 'stopped', 'message': 'Lỗi khởi động'})

@socketio.on('stop_counting')
def handle_stop_counting():
    """Dừng counting"""
    global current_counter, is_running
    
    try:
        if not is_running:
            logger.warning('Hệ thống chưa chạy!')
            return
        
        is_running = False
        if current_counter:
            current_counter.stop()
            current_counter = None
        
        emit('system_status', {'status': 'stopped', 'message': 'Đã dừng'})
        logger.success('Đã dừng hệ thống')
        
    except Exception as e:
        logger.error(f'Lỗi dừng hệ thống: {str(e)}')

@socketio.on('set_drawing_points')
def handle_set_drawing_points(data):
    """Thiết lập điểm vẽ cho line/zone"""
    global current_counter
    
    try:
        print(f"DEBUG: Received set_drawing_points: {data}")
        
        method = data.get('method')
        points = data.get('points', [])
        
        print(f"DEBUG: Method={method}, Points count={len(points)}")
        
        # Chuyển đổi points thành format phù hợp
        converted_points = [(point['x'], point['y']) for point in points]
        print(f"DEBUG: Converted points: {converted_points}")
        
        if method == 'line' and len(converted_points) >= 2:
            if current_counter and hasattr(current_counter, 'set_line_points'):
                current_counter.set_line_points(converted_points[:2])
                logger.success(f'Đã thiết lập đường đếm: {converted_points[:2]}')
            else:
                logger.error('Counter không hỗ trợ set_line_points')
        elif method == 'zone' and len(converted_points) >= 3:
            # Sử dụng ZoneCounterWeb set_zone_points
            if current_counter and hasattr(current_counter, 'set_zone_points'):
                current_counter.set_zone_points(converted_points)
                logger.success(f'Đã thiết lập vùng đếm: {len(converted_points)} điểm')
            else:
                logger.error('Counter không hỗ trợ set_zone_points')
        else:
            logger.error(f'Số điểm không hợp lệ cho {method}: {len(converted_points)}')
        
    except Exception as e:
        logger.error(f'Lỗi thiết lập drawing: {str(e)}')
        print(f"DEBUG: Exception in set_drawing_points: {e}")

@socketio.on('clear_drawing')
def handle_clear_drawing():
    """Xóa drawing hiện tại"""
    global current_counter
    
    try:
        print("DEBUG: Received clear_drawing")
        
        if current_counter:
            if hasattr(current_counter, 'clear_drawing'):
                current_counter.clear_drawing()
                logger.info('Đã xóa drawing')
            else:
                logger.error('Counter không hỗ trợ clear_drawing')
        else:
            logger.error('Không có counter đang chạy')
    except Exception as e:
        logger.error(f'Lỗi xóa drawing: {str(e)}')
        print(f"DEBUG: Exception in clear_drawing: {e}")

@socketio.on('reset_system')
def handle_reset_system():
    """Reset hệ thống"""
    global current_counter
    
    try:
        if current_counter:
            current_counter.reset()
        
        # Reset stats
        emit('stats_update', {
            'in_count': 0,
            'out_count': 0,
            'zone_count': 0
        })
        
        logger.success('Đã reset hệ thống')
        
    except Exception as e:
        logger.error(f'Lỗi reset: {str(e)}')

def run_counting():
    """Chạy counting trong thread riêng"""
    global current_counter, is_running
    
    try:
        print("🏃 run_counting() started...")  # Debug log
        if current_counter:
            print(f"✅ Found counter instance: {type(current_counter).__name__}")  # Debug log
            current_counter.run()
        else:
            print("❌ No counter instance found!")  # Debug log
    except Exception as e:
        print(f"❌ Error in run_counting: {e}")  # Debug log
        logger.error(f'Lỗi counting: {str(e)}')
        logger.error(f'Traceback: {traceback.format_exc()}')
    finally:
        is_running = False
        socketio.emit('system_status', {'status': 'stopped', 'message': 'Đã dừng'})

def encode_frame(frame):
    """Encode frame thành base64 để gửi qua Socket.IO"""
    try:
        # Lưu kích thước gốc
        original_height, original_width = frame.shape[:2]
        
        # Resize frame nếu quá lớn để tăng tốc độ
        display_frame = frame.copy()
        scale_factor = 1.0
        
        if original_width > 800:
            scale_factor = 800 / original_width
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            display_frame = cv2.resize(display_frame, (new_width, new_height))
        
        # Encode thành JPEG
        ret, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ret:
            frame_bytes = base64.b64encode(buffer).decode('utf-8')
            
            # Trả về cả frame và metadata
            return {
                'image': frame_bytes,
                'original_size': {
                    'width': original_width,
                    'height': original_height
                },
                'display_size': {
                    'width': display_frame.shape[1],
                    'height': display_frame.shape[0]
                },
                'scale_factor': scale_factor
            }
    except Exception as e:
        print(f"Encode frame error: {e}")
    return None

if __name__ == '__main__':
    print("="*60)
    print("🌐 PEOPLE COUNTING WEB SERVER")
    print("="*60)
    print("🚀 Starting Flask + Socket.IO server...")
    print("🌍 URL: http://localhost:5000")
    print("📱 Mobile access: http://<your-ip>:5000")
    print("="*60)
    
    try:
        # Tạo các web modules nếu chưa có
        if not os.path.exists('line_counting_web.py'):
            print("📝 Tạo line_counting_web.py...")
            # Code sẽ được tạo sau
        
        if not os.path.exists('zone_counting_new.py'):
            print("📝 Tạo zone_counting_new.py...")
            # Code sẽ được tạo sau
        
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
        
    except KeyboardInterrupt:
        print("\n⚠️ Server đã dừng")
    except Exception as e:
        print(f"\n❌ Lỗi server: {e}")
