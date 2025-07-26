#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zone Counting Web Module
Module đếm người trong vùng với streaming web
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import base64
from collections import defaultdict
import sys
import os

# Add paths for imports
current_dir = os.path.dirname(__file__)
app_dir = os.path.dirname(current_dir)
tracking_dir = os.path.join(app_dir, 'core', 'tracking')
config_dir = os.path.join(app_dir, '..', 'config')
core_dir = os.path.join(app_dir, 'core')

sys.path.insert(0, tracking_dir)
sys.path.insert(0, config_dir)
sys.path.insert(0, core_dir)

from deep_tracker import DeepTracker
from settings import CLASS_ID, CONF_THRESHOLD, get_model_path, get_video_path

# TensorRT support
try:
    from tensorrt_inference import create_yolo_model
    TENSORRT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ TensorRT not available: {e}")
    TENSORRT_AVAILABLE = False

class ZoneCounterWeb:
    def __init__(self, video_path, model_path, socketio, logger):
        self.video_path = video_path
        self.model_path = model_path
        self.socketio = socketio
        self.logger = logger
        
        # Khởi tạo
        self.cap = None
        self.model = None
        self.tracker = None
        self.running = False
        
        # Zone counting variables
        self.zone_points = []
        self.zone_ready = False
        self.current_in_zone = set()
        self.track_history = defaultdict(list)
        
        # Zone crossing tracking variables - để track người ra vào
        self.crossing_history = {}  # track_id -> previous zone status (True/False)
        self.people_entering = 0    # Số người vào zone
        self.people_leaving = 0     # Số người ra khỏi zone
        
        # Auto setup zone (giữa màn hình, hình chữ nhật) - sẽ bị ghi đè khi user vẽ
        self.auto_setup_zone = True
        self.user_defined_zone = False
        
    def initialize(self):
        """Khởi tạo camera/video và model"""
        try:
            # Mở video source
            if isinstance(self.video_path, int):
                self.logger.info(f'Mở camera ID: {self.video_path}')
                self.cap = cv2.VideoCapture(self.video_path)
            else:
                self.logger.info(f'Mở video file: {self.video_path}')
                self.cap = cv2.VideoCapture(self.video_path)
            
            if not self.cap.isOpened():
                raise Exception(f"Không thể mở video source: {self.video_path}")
            
            # Load YOLO model with TensorRT support
            self.logger.info(f'Loading model: {self.model_path}')
            
            if TENSORRT_AVAILABLE and self.model_path.endswith('.engine'):
                self.model = create_yolo_model('models/' + self.model_path, self.logger)
                self.logger.info(f'🚀 TensorRT engine loaded: {self.model_path}')
            else:
                self.model = YOLO('models/' + self.model_path)
                self.logger.info(f'📦 PyTorch model loaded: {self.model_path}')
            
            # Khởi tạo tracker
            self.tracker = DeepTracker(max_age=30, min_hits=3)
            
            # Auto setup zone nếu cần và chưa có user-defined zone
            if self.auto_setup_zone and not self.user_defined_zone:
                ret, frame = self.cap.read()
                if ret:
                    h, w = frame.shape[:2]
                    # Tạo zone hình chữ nhật ở giữa màn hình
                    margin_w = w // 4
                    margin_h = h // 4
                    self.zone_points = [
                        (margin_w, margin_h),          # Top-left
                        (w - margin_w, margin_h),      # Top-right  
                        (w - margin_w, h - margin_h),  # Bottom-right
                        (margin_w, h - margin_h)       # Bottom-left (không duplicate)
                    ]
                    self.zone_ready = True
                    self.logger.info(f'Tự động tạo vùng đếm: {len(self.zone_points)} điểm')
                    # Reset lại video
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            return True
            
        except Exception as e:
            self.logger.error(f'Lỗi khởi tạo: {str(e)}')
            return False
    
    def run(self):
        """Chạy zone counting"""
        if not self.initialize():
            return
        
        self.running = True
        self.logger.success('Bắt đầu Zone Counting')
        
        frame_count = 0
        fps_time = time.time()
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    if isinstance(self.video_path, str):  # Video file ended
                        self.logger.info('Video đã kết thúc, lặp lại...')
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:  # Camera error
                        self.logger.error('Lỗi đọc frame từ camera')
                        break
                
                frame_count += 1
                
                # Vẽ zone
                self.draw_zone(frame)
                
                # YOLO detection
                results = self.model.predict(source=frame, classes=[CLASS_ID], conf=CONF_THRESHOLD, verbose=False)
                detections = []
                
                if results and results[0].boxes.xyxy is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box[:4])
                        detections.append([x1, y1, x2, y2])
                
                # Tracking và zone counting
                active_tracks = self.tracker.update(detections, frame)
                
                # Đếm người trong zone - logic đơn giản như zone_counting.py
                current_frame_ids = set()
                for track in active_tracks:
                        x1, y1, x2, y2 = map(int, track.bbox)
                        cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)
                        
                        # Kiểm tra người có trong zone không
                        in_zone = False
                        if self.zone_ready and len(self.zone_points) > 2:
                            pts = np.array(self.zone_points, np.int32).reshape((-1, 1, 2))
                            in_zone = cv2.pointPolygonTest(pts, (cx, cy), False) >= 0
                        
                        # Zone crossing detection (giống line counting logic)
                        track_id = track.track_id
                        previous_status = self.crossing_history.get(track_id, None)
                        
                        if previous_status is not None:
                            if previous_status == False and in_zone == True:
                                # Person entered zone
                                self.people_entering += 1
                                self.logger.info(f"✅ Person {track_id} entered zone. Total entering: {self.people_entering}")
                            elif previous_status == True and in_zone == False:
                                # Person left zone
                                self.people_leaving += 1
                                self.logger.info(f"❌ Person {track_id} left zone. Total leaving: {self.people_leaving}")
                        
                        # Update crossing history
                        self.crossing_history[track_id] = in_zone
                        
                        if in_zone:
                            current_frame_ids.add(track.track_id)
                            # Vẽ người trong zone với màu đỏ
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, f"ID:{track.track_id}", (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            
                            # Lưu trajectory
                            self.track_history[track.track_id].append((cx, cy))
                            if len(self.track_history[track.track_id]) > 30:
                                self.track_history[track.track_id].pop(0)
                        else:
                            # Vẽ người ngoài zone với màu xanh dương
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(frame, f"ID:{track.track_id}", (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Clean up old crossing history cho tracks không còn active
                active_track_ids = {track.track_id for track in active_tracks}
                self.crossing_history = {tid: status for tid, status in self.crossing_history.items() 
                                       if tid in active_track_ids}
                
                # Cập nhật current_in_zone
                self.current_in_zone = current_frame_ids
                
                # Vẽ trajectories
                self.draw_trajectories(frame)
                
                # Vẽ stats trên frame
                self.draw_stats(frame)
                
                # Gửi frame qua Socket.IO
                self.send_frame(frame)
                
                # Gửi stats update
                self.send_stats()
                
                # FPS logging
                if frame_count % 60 == 0:
                    current_time = time.time()
                    fps = 60 / (current_time - fps_time) if current_time > fps_time else 0
                    fps_time = current_time
                    self.logger.info(f'FPS: {fps:.1f} | In Zone: {len(self.current_in_zone)} | Entering: {self.people_entering} | Leaving: {self.people_leaving}')
                
                # Tối ưu tốc độ streaming - giảm delay
                time.sleep(0.01)  # ~60 FPS target
                
        except Exception as e:
            self.logger.error(f'Lỗi trong quá trình counting: {str(e)}')
        finally:
            self.cleanup()
    
    def draw_zone(self, frame):
        """Vẽ vùng zone - chỉ đường viền như zone_counting.py"""
        if self.zone_ready and len(self.zone_points) > 2:
            # Vẽ đường viền polygon màu xanh lá
            pts = np.array(self.zone_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            
            # Vẽ các điểm góc với số thứ tự
            for i, point in enumerate(self.zone_points):
                cv2.circle(frame, point, 8, (0, 255, 0), -1)
                cv2.putText(frame, str(i+1), (point[0]+10, point[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    def draw_trajectories(self, frame):
        """Vẽ trajectories của các track"""
        for track_id, points in self.track_history.items():
            if len(points) > 1:
                for i in range(1, len(points)):
                    cv2.line(frame, points[i-1], points[i], (255, 0, 255), 2)
    
    def draw_stats(self, frame):
        """Vẽ thống kê trên frame với entering/leaving counts"""
        # Background box
        cv2.rectangle(frame, (10, 10), (300, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 150), (255, 255, 255), 2)
        
        # Text
        cv2.putText(frame, "ZONE COUNTING", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"In Zone: {len(self.current_in_zone)}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Entering: {self.people_entering}", (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Leaving: {self.people_leaving}", (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    def send_frame(self, frame):
        """Gửi frame qua Socket.IO với metadata về kích thước - Tối ưu hóa"""
        try:
            # Lưu kích thước gốc
            original_height, original_width = frame.shape[:2]
            
            # Aggressive resize để tăng tốc
            display_frame = frame.copy()
            scale_factor = 1.0
            
            # Resize xuống kích thước nhỏ hơn để tăng tốc
            if original_width > 640:  # Giảm từ 800 xuống 640
                scale_factor = 640 / original_width
                new_width = int(original_width * scale_factor)
                new_height = int(original_height * scale_factor)
                display_frame = cv2.resize(display_frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # Tối ưu JPEG compression
            encode_params = [
                cv2.IMWRITE_JPEG_QUALITY, 60,  # Giảm từ 70 xuống 60
                cv2.IMWRITE_JPEG_OPTIMIZE, 1,
                cv2.IMWRITE_JPEG_PROGRESSIVE, 1
            ]
            ret, buffer = cv2.imencode('.jpg', display_frame, encode_params)
            
            if ret:
                frame_bytes = base64.b64encode(buffer).decode('utf-8')
                
                # Gửi với metadata
                frame_data = {
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
                self.socketio.emit('video_frame', frame_data)
        except Exception as e:
            print(f"Send frame error: {e}")
    
    def send_stats(self):
        """Gửi stats qua Socket.IO với entering/leaving counts"""
        try:
            stats_data = {
                'in_count': self.people_entering,      # Số người vào zone
                'out_count': self.people_leaving,      # Số người ra khỏi zone
                'zone_count': len(self.current_in_zone)  # Số người hiện tại trong zone
            }
            self.socketio.emit('stats_update', stats_data)
        except Exception as e:
            print(f"Send stats error: {e}")
    
    def stop(self):
        """Dừng counting"""
        self.running = False
        # Cleanup TensorRT resources
        if hasattr(self.model, 'cleanup'):
            self.model.cleanup()
        self.logger.info('Zone Counting đã dừng')
    
    def reset(self):
        """Reset counting"""
        self.current_in_zone = set()
        self.track_history = defaultdict(list)
        self.crossing_history = {}  # Reset crossing history
        self.people_entering = 0    # Reset entering count
        self.people_leaving = 0     # Reset leaving count
        if self.tracker:
            self.tracker = DeepTracker(max_age=30, min_hits=3)
        self.logger.info('Đã reset Zone Counting')
    
    def cleanup(self):
        """Dọn dẹp resources"""
        if self.cap:
            self.cap.release()
        # Cleanup TensorRT resources
        if hasattr(self.model, 'cleanup'):
            self.model.cleanup()
        self.logger.info('Đã dọn dẹp resources')
    
    def set_zone_points(self, points):
        """Thiết lập điểm zone từ web interface"""
        if len(points) >= 3:
            self.zone_points = points
            self.zone_ready = True
            self.user_defined_zone = True
            self.logger.info(f'Đã thiết lập zone points: {len(points)} điểm')
    
    def clear_drawing(self):
        """Xóa drawing"""
        self.zone_points = []
        self.zone_ready = False
        self.user_defined_zone = False
        self.logger.info('Đã xóa zone drawing')

if __name__ == "__main__":
    # Test standalone
    class DummySocketIO:
        def emit(self, event, data):
            print(f"Emit {event}: {data}")
    
    class DummyLogger:
        def info(self, msg): print(f"INFO: {msg}")
        def success(self, msg): print(f"SUCCESS: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
    
    counter = ZoneCounterWeb('Test.mp4', 'yolov8s.pt', DummySocketIO(), DummyLogger())
    counter.run()
