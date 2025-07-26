import cv2
import numpy as np
import os
import warnings

# Tắt YOLO logging hoàn toàn
os.environ['YOLO_VERBOSE'] = 'False'
os.environ['ULTRALYTICS_VERBOSE'] = 'False'  
warnings.filterwarnings("ignore")

from ultralytics import YOLO
import torch
from collections import defaultdict, deque
from torchvision import models, transforms
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import time
import base64

# CNN Feature Extractor (giống line counting)
class FeatureExtractor:
    def __init__(self):
        # Tắt deprecation warnings
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        
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
        return features.cpu().numpy().reshape(len(batch), -1)

class Track:
    def __init__(self, track_id, bbox, feature):
        self.track_id = track_id
        self.bbox = bbox
        self.feature = feature
        self.hits = 1
        self.time_since_update = 0
        self.history = []
        self.age = 0

    def update(self, bbox, feature):
        self.bbox = bbox
        self.feature = feature
        self.hits += 1
        self.time_since_update = 0
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        self.history.append(center)
        if len(self.history) > 30:
            self.history.pop(0)

    def predict(self):
        self.time_since_update += 1
        self.age += 1

    def to_xyxy(self):
        return self.bbox

# Tracker giống line counting
class DeepTracker:
    def __init__(self, max_age=30, min_hits=3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self.track_id_counter = 1
        self.feature_extractor = FeatureExtractor()

    def _calculate_cost_matrix(self, detections, features):
        if len(self.tracks) == 0 or len(detections) == 0:
            return np.array([])
        
        # Position distance
        track_centers = np.array([[(t.bbox[0] + t.bbox[2]) / 2, (t.bbox[1] + t.bbox[3]) / 2] for t in self.tracks])
        det_centers = np.array([[(d[0] + d[2]) / 2, (d[1] + d[3]) / 2] for d in detections])
        position_cost = cdist(track_centers, det_centers, metric='euclidean')
        position_cost /= np.max(position_cost) if np.max(position_cost) > 0 else 1
        
        # Feature distance
        track_features = np.array([t.feature for t in self.tracks])
        feature_cost = cdist(track_features, features, metric='cosine')
        
        # Combined cost
        total_cost = 0.3 * position_cost + 0.7 * feature_cost
        return total_cost

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

# Global variables
tracking_enabled = True  # Bật tracking mặc định
zone_points = []
people_count = 0
people_entering = 0
people_leaving = 0

def set_zone_points(points):
    """Set zone points từ web interface"""
    global zone_points
    zone_points = points
    print(f"✅ Zone points set: {len(points)} points")

def toggle_tracking():
    """Toggle tracking on/off"""
    global tracking_enabled
    tracking_enabled = not tracking_enabled
    print(f"🎯 Tracking toggled: {'ON' if tracking_enabled else 'OFF'}")
    return tracking_enabled

def get_stats():
    """Get current stats"""
    return {
        'current_count': people_count,
        'entering': people_entering,
        'leaving': people_leaving,
        'tracking_enabled': tracking_enabled
    }

def run_zone_counting(socketio, video_path=None):
    global tracking_enabled, zone_points, people_count, people_entering, people_leaving
    
    # Đảm bảo tracking được bật từ đầu
    tracking_enabled = True
    print(f"🎯 Zone counting started with tracking_enabled = {tracking_enabled}")
    
    cap = cv2.VideoCapture(video_path if video_path else 0)
    if not cap.isOpened():
        print("Error: Could not open video source")
        return

    # Load YOLO model (tắt verbose logging hoàn toàn)
    import warnings
    import logging
    warnings.filterwarnings("ignore")
    logging.getLogger('ultralytics').setLevel(logging.WARNING)
    
    model = YOLO('yolov8m.pt', verbose=False)
    tracker = DeepTracker()
    
    # Zone crossing tracking variables (giống line counting)
    crossing_history = {}  # track_id -> previous zone status (True/False)
    frame_count = 0
    fps_start_time = time.time()
    fps_count = 0
    current_fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            if video_path and isinstance(video_path, str):  # Video file
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
                continue
            else:  # Camera or error
                break

        # Đảm bảo frame không None và có kích thước hợp lệ
        if frame is None or frame.size == 0:
            continue

        frame_count += 1
        fps_count += 1
        
        # FPS calculation
        if fps_count >= 30:
            current_fps = fps_count / (time.time() - fps_start_time)
            fps_start_time = time.time()
            fps_count = 0

        # Detect người (tắt verbose logging)
        results = model(frame, classes=[0], conf=0.5, verbose=False)
        detections = []
        
        # Kiểm tra results có tồn tại và có boxes
        if results and len(results) > 0:
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        detections.append([x1, y1, x2, y2, conf])

        # Track
        if tracking_enabled:
            tracks = tracker.update(detections, frame)
            
            # Zone counting logic với crossing detection
            if len(zone_points) >= 3:
                zone_poly = np.array(zone_points, np.int32)
                current_in_zone = 0
                
                for track in tracks:
                    x1, y1, x2, y2 = track.to_xyxy()
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # Check if person is in zone
                    in_zone = cv2.pointPolygonTest(zone_poly, (center_x, center_y), False) >= 0
                    
                    if in_zone:
                        current_in_zone += 1
                    
                    # Track zone crossings (giống line crossing logic)
                    track_id = track.track_id
                    previous_status = crossing_history.get(track_id, None)
                    
                    if previous_status is not None:
                        if previous_status == False and in_zone == True:
                            # Person entered zone
                            people_entering += 1
                            print(f"✅ Person {track_id} entered zone. Total entering: {people_entering}")
                        elif previous_status == True and in_zone == False:
                            # Person left zone
                            people_leaving += 1
                            print(f"❌ Person {track_id} left zone. Total leaving: {people_leaving}")
                    
                    crossing_history[track_id] = in_zone
                
                # Clean up old crossing history
                active_track_ids = {track.track_id for track in tracks}
                crossing_history = {tid: status for tid, status in crossing_history.items() 
                                  if tid in active_track_ids}
                
                people_count = current_in_zone

            # Draw tracking boxes
            for track in tracks:
                x1, y1, x2, y2 = track.to_xyxy()
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'ID: {track.track_id}', (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # Draw detection boxes khi không track
            for det in detections:
                x1, y1, x2, y2 = det[:4]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        # Draw zone
        if len(zone_points) >= 3:
            zone_poly = np.array(zone_points, np.int32)
            
            # Tạo overlay để vẽ zone với độ trong suốt
            overlay = frame.copy()
            cv2.fillPoly(overlay, [zone_poly], color=(0, 0, 255))  # Màu đỏ BGR
            
            # Blend overlay với frame gốc (độ trong suốt 30%)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # Vẽ viền zone màu đỏ
            cv2.polylines(frame, [zone_poly], isClosed=True, color=(0, 0, 255), thickness=3)
        
        # Draw in-progress zone
        if len(zone_points) >= 2:
            for i in range(len(zone_points) - 1):
                cv2.line(frame, tuple(zone_points[i]), tuple(zone_points[i + 1]), (0, 255, 255), 2)
        
        # Mark zone points
        for i, point in enumerate(zone_points):
            cv2.circle(frame, tuple(point), 5, (255, 255, 0), -1)
            cv2.putText(frame, str(i + 1), (point[0] + 10, point[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Draw stats với background để dễ đọc
        stats_y = 30
        cv2.rectangle(frame, (10, 10), (350, 160), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, 160), (255, 255, 255), 2)
        
        cv2.putText(frame, f'Trong zone: {people_count}', (20, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        stats_y += 30
        cv2.putText(frame, f'Vao: {people_entering}', (20, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        stats_y += 30
        cv2.putText(frame, f'Ra: {people_leaving}', (20, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        stats_y += 30
        cv2.putText(frame, f'FPS: {current_fps:.1f}', (20, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        stats_y += 30
        cv2.putText(frame, f'Tracking: {"ON" if tracking_enabled else "OFF"}', (20, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if tracking_enabled else (0, 0, 255), 2)

        # Encode và emit frame
        try:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if buffer is not None and len(buffer) > 0:
                frame_data = base64.b64encode(buffer).decode('utf-8')
                
                original_height, original_width = frame.shape[:2]
                socketio.emit('video_frame', {
                    'frame': frame_data,
                    'stats': {
                        'current_count': people_count,
                        'entering': people_entering,
                        'leaving': people_leaving,
                        'fps': current_fps,
                        'tracking_enabled': tracking_enabled
                    },
                    'frame_metadata': {
                        'original_width': original_width,
                        'original_height': original_height
                    }
                })
        except Exception as e:
            print(f"Error encoding frame: {e}")
            continue
        
        time.sleep(0.01)  # Tối thiểu delay cho smooth streaming

    cap.release()
