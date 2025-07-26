#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Line Counting Web Module
Module đếm người qua đường thẳng với streaming web
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
from torchvision import models
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import time
import base64
import threading
from collections import deque
import sys
import os

# Add paths for TensorRT support
current_dir = os.path.dirname(__file__)
core_dir = os.path.join(current_dir, '..', 'core')
sys.path.insert(0, core_dir)

try:
    from tensorrt_inference import create_yolo_model
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    create_yolo_model = None

# Cấu hình
CLASS_ID = 0  # ID cho class 'person'
CONF_THRESHOLD = 0.5

# CNN Feature Extractor
class FeatureExtractor:
    def __init__(self):
        self.model = models.resnet18(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def extract_features(self, image_patches):
        if len(image_patches) == 0:
            return np.array([])
        batch = []
        for patch in image_patches:
            if patch.size > 0:
                tensor = self.transform(patch)
                batch.append(tensor)
        if len(batch) == 0:
            return np.array([])
        batch = torch.stack(batch)
        if torch.cuda.is_available():
            batch = batch.cuda()
        with torch.no_grad():
            features = self.model(batch)
            features = features.squeeze()
            if len(features.shape) == 1:
                features = features.unsqueeze(0)
            features = features.cpu().numpy()
        return features

class Track:
    def __init__(self, track_id, detection, feature):
        self.track_id = track_id
        self.hits = 1
        self.time_since_update = 0
        self.feature = feature
        self.positions = []
        x1, y1, x2, y2 = detection[:4]
        self.x = (x1 + x2) / 2
        self.y = (y1 + y2) / 2
        self.positions.append((self.x, self.y))
        self.bbox = detection[:4]
        self.crossing_history = deque(maxlen=5)
        self.counted = False

    def update(self, detection, feature):
        self.hits += 1
        self.time_since_update = 0
        self.feature = 0.8 * self.feature + 0.2 * feature
        x1, y1, x2, y2 = detection[:4]
        self.x = (x1 + x2) / 2
        self.y = (y1 + y2) / 2
        self.positions.append((self.x, self.y))
        if len(self.positions) > 5:
            self.positions.pop(0)
        self.bbox = detection[:4]

    def predict(self):
        self.time_since_update += 1

class DeepTracker:
    def __init__(self, max_age=30, min_hits=3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self.track_id_counter = 1
        self.feature_extractor = FeatureExtractor()
        # Không skip frames cho accuracy tối đa

    def update(self, detections, frame):
        if len(detections) == 0:
            # Clean up old tracks
            self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]
            for track in self.tracks:
                track.predict()
            return []
        
        # Extract features cho mọi detection
        patches = []
        for det in detections:
            x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            patch = frame[y1:y2, x1:x2] 
            patches.append(patch)
        features = self.feature_extractor.extract_features(patches)
        cost_matrix = self._calculate_cost_matrix(detections, features)
        
        if len(self.tracks) > 0 and cost_matrix.size > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            matched_tracks = set()
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] < 0.7:
                    self.tracks[row].update(detections[col], features[col])
                    matched_tracks.add(row)
            for i, det in enumerate(detections):
                if i not in col_indices or cost_matrix[row_indices[list(col_indices).index(i)], i] >= 0.7:
                    new_track = Track(self.track_id_counter, det, features[i])
                    self.tracks.append(new_track)
                    self.track_id_counter += 1
            self.tracks = [t for i, t in enumerate(self.tracks) 
                           if i in matched_tracks or t.time_since_update < self.max_age]
        else:
            for i, det in enumerate(detections):
                new_track = Track(self.track_id_counter, det, features[i])
                self.tracks.append(new_track)
                self.track_id_counter += 1
        return [t for t in self.tracks if t.hits >= self.min_hits]

    def _calculate_cost_matrix(self, detections, features):
        if len(self.tracks) == 0 or len(detections) == 0:
            return np.array([]).reshape(0, 0)
        track_features = np.array([t.feature for t in self.tracks])
        feature_distances = cdist(track_features, features, metric='cosine')
        track_positions = np.array([[t.x, t.y] for t in self.tracks])
        det_positions = np.array([[d[0] + d[2]/2, d[1] + d[3]/2] for d in detections])
        position_distances = cdist(track_positions, det_positions, metric='euclidean')
        position_distances = position_distances / np.max(position_distances) if np.max(position_distances) > 0 else position_distances
        return 0.7 * feature_distances + 0.3 * position_distances

class LineCounterWeb:
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
        
        # Line counting variables
        self.line_points = []
        self.line_ready = False
        self.counts = {'IN': 0, 'OUT': 0}
        
        # Auto setup line (giữa màn hình, ngang) - sẽ bị ghi đè khi user vẽ
        self.auto_setup_line = True
        self.user_defined_line = False
        
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
            
            # Auto setup line nếu cần và chưa có user-defined line
            if self.auto_setup_line and not self.user_defined_line:
                ret, frame = self.cap.read()
                if ret:
                    h, w = frame.shape[:2]
                    # Tạo đường ngang ở giữa màn hình
                    self.line_points = [
                        (w // 4, h // 2),
                        (3 * w // 4, h // 2)
                    ]
                    self.line_ready = True
                    self.logger.info(f'Tự động tạo đường đếm: {self.line_points}')
                    # Reset lại video
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            return True
            
        except Exception as e:
            self.logger.error(f'Lỗi khởi tạo: {str(e)}')
            return False
    
    def run(self):
        """Chạy line counting"""
        if not self.initialize():
            return
        
        self.running = True
        self.logger.success('Bắt đầu Line Counting')
        
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
                
                # Vẽ đường đếm
                self.draw_line(frame)
                
                # YOLO detection - chạy trên mọi frame
                results = self.model(frame, classes=[CLASS_ID], conf=CONF_THRESHOLD, verbose=False)
                detections = []
                
                if len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()
                    for box, conf in zip(boxes, confidences):
                        x1, y1, x2, y2 = box
                        detections.append([x1, y1, x2, y2, conf])
                
                # Tracking
                active_tracks = self.tracker.update(detections, frame)
                
                # Kiểm tra crossing và vẽ tracking
                for track in active_tracks:
                    direction = self.check_crossing(track)
                    x1, y1, x2, y2 = track.bbox
                    
                    # Màu box dựa trên crossing
                    color = (0, 255, 255) if direction else (255, 0, 0)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.circle(frame, (int(track.x), int(track.y)), 5, color, -1)
                    
                    # Label
                    label = f"ID:{track.track_id}"
                    if direction:
                        label += f" {direction}"
                    cv2.putText(frame, label, (int(x1), int(y1)-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Vẽ trajectory
                    if len(track.positions) > 1:
                        for i in range(1, len(track.positions)):
                            pt1 = (int(track.positions[i-1][0]), int(track.positions[i-1][1]))
                            pt2 = (int(track.positions[i][0]), int(track.positions[i][1]))
                            cv2.line(frame, pt1, pt2, (128, 128, 128), 2)
                
                # Vẽ stats trên frame
                self.draw_stats(frame)
                
                # Gửi frame qua Socket.IO
                self.send_frame(frame)
                
                # Gửi stats update
                self.send_stats()
                
                # FPS logging với thông tin TensorRT
                if frame_count % 60 == 0:
                    current_time = time.time()
                    fps = 60 / (current_time - fps_time) if current_time > fps_time else 0
                    fps_time = current_time
                    model_type = "🚀 TensorRT" if self.model_path.endswith('.engine') else "📦 PyTorch"
                    self.logger.info(f'{model_type} FPS: {fps:.1f} | Tracks: {len(active_tracks)}')
                
                # Tối ưu tốc độ streaming - giảm delay
                time.sleep(0.01)  # ~60 FPS target
                
        except Exception as e:
            self.logger.error(f'Lỗi trong quá trình counting: {str(e)}')
        finally:
            self.cleanup()
    
    def check_crossing(self, track):
        """Kiểm tra crossing qua line"""
        if track.counted or not self.line_ready or len(self.line_points) != 2 or len(track.positions) < 2:
            return None
        
        prev = track.positions[-2]
        curr = track.positions[-1]
        
        # Kiểm tra line intersection
        intersects, _ = self.line_intersection(prev, curr, self.line_points[0], self.line_points[1])
        
        if intersects:
            direction = self.get_crossing_direction(prev, curr, self.line_points[0], self.line_points[1])
            track.crossing_history.append(direction)
        else:
            track.crossing_history.append(None)
        
        # Xác định crossing
        history = list(track.crossing_history)
        if history.count("IN") >= 1:
            self.counts["IN"] += 1
            track.counted = True
            self.logger.success(f'Người #{track.track_id} vào | IN = {self.counts["IN"]}')
            return "IN"
        elif history.count("OUT") >= 1:
            self.counts["OUT"] += 1
            track.counted = True
            self.logger.success(f'Người #{track.track_id} ra | OUT = {self.counts["OUT"]}')
            return "OUT"
        
        return None
    
    def line_intersection(self, p1, p2, q1, q2):
        """Kiểm tra intersection giữa 2 đường thẳng"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = q1
        x4, y4 = q2
        
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-10:
            return False, None
        
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
        u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            ix = x1 + t*(x2 - x1)
            iy = y1 + t*(y2 - y1)
            return True, (ix, iy)
        
        return False, None
    
    def get_crossing_direction(self, prev, curr, line_p1, line_p2):
        """Xác định hướng crossing"""
        dx, dy = curr[0] - prev[0], curr[1] - prev[1]
        lx, ly = line_p2[0] - line_p1[0], line_p2[1] - line_p1[1]
        cross = dx * ly - dy * lx
        return "IN" if cross > 0 else "OUT"
    
    def draw_line(self, frame):
        """Vẽ đường đếm"""
        if len(self.line_points) == 2:
            cv2.line(frame, self.line_points[0], self.line_points[1], (0, 0, 255), 3)
            cv2.circle(frame, self.line_points[0], 8, (0, 255, 0), -1)
            cv2.circle(frame, self.line_points[1], 8, (0, 255, 0), -1)
            
            # Vẽ label IN/OUT
            mid_x = (self.line_points[0][0] + self.line_points[1][0]) // 2
            mid_y = (self.line_points[0][1] + self.line_points[1][1]) // 2
            cv2.putText(frame, "IN", (mid_x - 50, mid_y - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "OUT", (mid_x + 20, mid_y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def draw_stats(self, frame):
        """Vẽ thống kê trên frame"""
        # Background box
        cv2.rectangle(frame, (10, 10), (300, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 100), (255, 255, 255), 2)
        
        # Text
        cv2.putText(frame, "LINE COUNTING", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"IN:  {self.counts['IN']}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"OUT: {self.counts['OUT']}", (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        net = self.counts['IN'] - self.counts['OUT']
        cv2.putText(frame, f"NET: {net}", (150, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
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
        """Gửi stats qua Socket.IO"""
        try:
            self.socketio.emit('stats_update', {
                'in_count': self.counts['IN'],
                'out_count': self.counts['OUT'],
                'zone_count': 0  # Line counting không có zone count
            })
        except Exception as e:
            print(f"Send stats error: {e}")
    
    def stop(self):
        """Dừng counting"""
        self.running = False
        # Cleanup TensorRT resources
        if hasattr(self.model, 'cleanup'):
            self.model.cleanup()
        self.logger.info('Line Counting đã dừng')
    
    def reset(self):
        """Reset counting"""
        self.counts = {'IN': 0, 'OUT': 0}
        if self.tracker:
            self.tracker = DeepTracker(max_age=30, min_hits=3)
        self.logger.info('Đã reset Line Counting')
    
    def cleanup(self):
        """Dọn dẹp resources"""
        if self.cap:
            self.cap.release()
        # Cleanup TensorRT resources
        if hasattr(self.model, 'cleanup'):
            self.model.cleanup()
        self.logger.info('Đã dọn dẹp resources')
    
    def set_line_points(self, points):
        """Thiết lập điểm line từ web interface"""
        if len(points) >= 2:
            self.line_points = points[:2]
            self.line_ready = True
            self.user_defined_line = True
            self.logger.info(f'Đã thiết lập line points: {self.line_points}')
    
    def clear_drawing(self):
        """Xóa drawing"""
        self.line_points = []
        self.line_ready = False
        self.user_defined_line = False
        self.logger.info('Đã xóa line drawing')

if __name__ == "__main__":
    # Test standalone
    class DummySocketIO:
        def emit(self, event, data):
            print(f"Emit {event}: {data}")
    
    class DummyLogger:
        def info(self, msg): print(f"INFO: {msg}")
        def success(self, msg): print(f"SUCCESS: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
    
    counter = LineCounterWeb('Test.mp4', 'yolov8s.pt', DummySocketIO(), DummyLogger())
    counter.run()
