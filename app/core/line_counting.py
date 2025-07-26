import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
from torchvision import models
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import time
from collections import deque

# Cấu hình
CLASS_ID = 0  # ID cho class 'person'
CONF_THRESHOLD = 0.5

# Biến global cho counting
line_points = []
line_ready = False
counts = {'IN': 0, 'OUT': 0}

# CNN Feature Extractor cho tracking
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

    def update(self, detections, frame):
        if len(detections) == 0:
            self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]
            return []
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

def line_intersection(p1, p2, q1, q2):
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

def get_crossing_direction(prev, curr, line_p1, line_p2):
    dx, dy = curr[0] - prev[0], curr[1] - prev[1]
    lx, ly = line_p2[0] - line_p1[0], line_p2[1] - line_p1[1]
    cross = dx * ly - dy * lx
    return "IN" if cross > 0 else "OUT"

def check_crossing(track):
    global counts
    if track.counted or not line_ready or len(line_points) != 2 or len(track.positions) < 2:
        return None
    prev = track.positions[-2]
    curr = track.positions[-1]
    intersects, _ = line_intersection(prev, curr, line_points[0], line_points[1])
    if intersects:
        direction = get_crossing_direction(prev, curr, line_points[0], line_points[1])
        track.crossing_history.append(direction)
    else:
        track.crossing_history.append(None)
    history = list(track.crossing_history)
    if history.count("IN") >= 1:
        counts["IN"] += 1
        track.counted = True
        print(f"✅ Người #{track.track_id} vào | IN = {counts['IN']}")
        return "IN"
    elif history.count("OUT") >= 1:
        counts["OUT"] += 1
        track.counted = True
        print(f"✅ Người #{track.track_id} ra | OUT = {counts['OUT']}")
        return "OUT"
    return None

def draw_line(frame):
    if len(line_points) == 1:
        cv2.circle(frame, line_points[0], 5, (0, 255, 0), -1)
    elif len(line_points) == 2:
        cv2.line(frame, line_points[0], line_points[1], (0, 0, 255), 3)
        cv2.circle(frame, line_points[0], 8, (0, 255, 0), -1)
        cv2.circle(frame, line_points[1], 8, (0, 255, 0), -1)
        mid_x = (line_points[0][0] + line_points[1][0]) // 2
        mid_y = (line_points[0][1] + line_points[1][1]) // 2
        cv2.putText(frame, "IN", (mid_x - 50, mid_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "OUT", (mid_x + 20, mid_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def draw_stats(frame):
    cv2.rectangle(frame, (10, 10), (250, 100), (0, 0, 0), -1)
    cv2.putText(frame, "CNN PEOPLE COUNTER", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"IN:  {counts['IN']}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"OUT: {counts['OUT']}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    net = counts['IN'] - counts['OUT']
    cv2.putText(frame, f"NET: {net}", (130, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

def reset_system():
    global line_points, line_ready, counts
    line_points.clear()
    line_ready = False
    counts = {'IN': 0, 'OUT': 0}
    print("Đã reset hệ thống!")

def mouse_callback(event, x, y, flags, param):
    global line_points, line_ready
    if event == cv2.EVENT_LBUTTONDOWN and len(line_points) < 2:
        line_points.append((x, y))
        if len(line_points) == 2:
            line_ready = True
            print(f"Đã vẽ đường đếm: {line_points}")

def line_counter(video_path='Test.mp4', model='yolov8l.pt'):
    global line_points, line_ready
    print("=== CNN DEEP TRACKING PEOPLE COUNTER ===")
    print("Sử dụng ResNet18 features để tracking chính xác")
    
    # Xác định nguồn video (file hoặc camera)
    if isinstance(video_path, int):
        print(f"📹 Đang sử dụng camera ID: {video_path}")
        cap = cv2.VideoCapture(video_path)
    else:
        print(f"📁 Đang sử dụng video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
    
    model = YOLO(model)
    tracker = DeepTracker(max_age=30, min_hits=3)
    
    if not cap.isOpened():
        if isinstance(video_path, int):
            print(f"❌ Không thể mở camera ID: {video_path}")
        else:
            print(f"❌ Không thể mở video file: {video_path}")
        return
    cv2.namedWindow('CNN People Counter')
    cv2.setMouseCallback('CNN People Counter', mouse_callback)
    print("Hướng dẫn:")
    print("1. Click 2 điểm để vẽ đường đếm")
    print("2. Nhấn 'r' để reset")
    print("3. Nhấn 'q' để thoát")
    if isinstance(video_path, int):
        print("4. Đang sử dụng camera - đảm bảo camera được kết nối")
    frame_count = 0
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        draw_line(frame)
        results = model(frame, classes=[CLASS_ID], conf=CONF_THRESHOLD, verbose=False)
        detections = []
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            for box, conf in zip(boxes, confidences):
                x1, y1, x2, y2 = box
                detections.append([x1, y1, x2, y2, conf])
        active_tracks = tracker.update(detections, frame)
        for track in active_tracks:
            direction = check_crossing(track)
            x1, y1, x2, y2 = track.bbox
            color = (0, 255, 255) if direction else (255, 0, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.circle(frame, (int(track.x), int(track.y)), 5, color, -1)
            label = f"ID:{track.track_id}"
            if direction:
                label += f" {direction}"
            cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if len(track.positions) > 1:
                for i in range(1, len(track.positions)):
                    pt1 = (int(track.positions[i-1][0]), int(track.positions[i-1][1]))
                    pt2 = (int(track.positions[i][0]), int(track.positions[i][1]))
                    cv2.line(frame, pt1, pt2, (128, 128, 128), 2)
        draw_stats(frame)
        if not line_ready:
            cv2.putText(frame, "Click 2 points to draw counting line", 
                        (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if frame_count % 30 == 0:
            runtime = time.time() - start_time
            fps = frame_count / runtime if runtime > 0 else 0
            print(f"FPS: {fps:.1f} | Active tracks: {len(active_tracks)}")
        cv2.putText(frame, f"CNN Tracking | Frame: {frame_count}", 
                    (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imshow('CNN People Counter', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            reset_system()
            tracker = DeepTracker(max_age=30, min_hits=3)
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n=== KẾT QUẢ CUỐI CÙNG ===")
    if isinstance(video_path, int):
        print(f"📹 Nguồn: Camera ID {video_path}")
    else:
        print(f"📁 Nguồn: {video_path}")
    print(f"Vào: {counts['IN']}")
    print(f"Ra: {counts['OUT']}")
    print(f"Ròng: {counts['IN'] - counts['OUT']}")
    print(f"Tổng frames: {frame_count}")

if __name__ == "__main__":
    line_counter()
