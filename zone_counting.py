import cv2
import numpy as np
from ultralytics import YOLO
import torch
from collections import defaultdict, deque
from torchvision import models, transforms
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# --- THIẾT LẬP ---
CLASS_ID = 0  # 'person'
CONF_THRESHOLD = 0.5

# --- VÙNG ZONE ---
def draw_zone(event, x, y, flags, param):
    zone_points, zone_ready = param
    if event == cv2.EVENT_LBUTTONDOWN and not zone_ready[0]:
        zone_points.append((x, y))
        print(f"Point {len(zone_points)}: ({x}, {y})")
        if len(zone_points) == 5:
            zone_ready[0] = True
            print("Zone is ready!")

# --- TRÍCH XUẤT ĐẶC TRƯNG CNN ---
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

# --- ĐỊNH NGHĨA TRACK ---
class Track:
    def __init__(self, track_id, detection, feature):
        self.track_id = track_id
        self.hits = 1
        self.time_since_update = 0
        self.feature = feature
        x1, y1, x2, y2 = detection[:4]
        self.x = (x1 + x2) / 2
        self.y = (y1 + y2) / 2
        self.positions = [(self.x, self.y)]
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

# --- TRÌNH TRACKER CHÍNH ---
class DeepTracker:
    def __init__(self, max_age=30, min_hits=3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self.track_id_counter = 1
        self.feature_extractor = FeatureExtractor()

    def update(self, detections, frame):
        if len(detections) == 0:
            for t in self.tracks:
                t.predict()
            self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]
            return []

        patches = []
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            patch = frame[y1:y2, x1:x2]
            patches.append(patch)

        features = self.feature_extractor.extract_features(patches)
        cost_matrix = self._calculate_cost_matrix(detections, features)

        matched_tracks = set()
        unmatched_detections = set(range(len(detections)))

        if len(self.tracks) > 0 and cost_matrix.size > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] < 0.7:
                    self.tracks[row].update(detections[col], features[col])
                    matched_tracks.add(row)
                    unmatched_detections.discard(col)

        for i in unmatched_detections:
            new_track = Track(self.track_id_counter, detections[i], features[i])
            self.tracks.append(new_track)
            self.track_id_counter += 1

        for t in self.tracks:
            t.predict()

        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]
        return [t for t in self.tracks if t.hits >= self.min_hits]

    def _calculate_cost_matrix(self, detections, features):
        if len(self.tracks) == 0 or len(detections) == 0:
            return np.array([]).reshape(0, 0)
        track_features = np.array([t.feature for t in self.tracks])
        feature_distances = cdist(track_features, features, metric='cosine')
        track_positions = np.array([[t.x, t.y] for t in self.tracks])
        det_positions = np.array([[d[0] + (d[2] - d[0]) / 2, d[1] + (d[3] - d[1]) / 2] for d in detections])
        position_distances = cdist(track_positions, det_positions, metric='euclidean')
        if np.max(position_distances) > 0:
            position_distances = position_distances / np.max(position_distances)
        return 0.7 * feature_distances + 0.3 * position_distances

# --- CHƯƠNG TRÌNH CHÍNH ---
def zone_counter(video_path='Test.mp4', model='yolov9s.pt'):
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

    model = YOLO(model).to('cuda:0')
    tracker = DeepTracker()

    zone_points = []
    zone_ready = [False]
    current_in_zone = set()
    track_history = defaultdict(list)

    # Xác định nguồn video (file hoặc camera)
    if isinstance(video_path, int):
        print(f"📹 Đang sử dụng camera ID: {video_path}")
        cap = cv2.VideoCapture(video_path)
    else:
        print(f"📁 Đang sử dụng video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
    if not cap.isOpened():
        if isinstance(video_path, int):
            print(f"❌ Không thể mở camera ID: {video_path}")
        else:
            print(f"❌ Không thể mở video file: {video_path}")
        return
        
    cv2.namedWindow('Zone People Counter')
    cv2.setMouseCallback('Zone People Counter', draw_zone, param=(zone_points, zone_ready))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        for i, point in enumerate(zone_points):
            cv2.circle(frame, point, 8, (0, 255, 0), -1)
            cv2.putText(frame, str(i+1), (point[0]+10, point[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if zone_ready[0]:
            pts = np.array(zone_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        else:
            pts = None

        results = model.predict(source=frame, classes=[CLASS_ID], conf=CONF_THRESHOLD)
        detections = []
        if results and results[0].boxes.xyxy is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                detections.append([x1, y1, x2, y2])

        active_tracks = tracker.update(detections, frame)

        current_frame_ids = set()
        for track in active_tracks:
            x1, y1, x2, y2 = map(int, track.bbox)
            cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)
            if zone_ready[0] and cv2.pointPolygonTest(pts, (cx, cy), False) >= 0:
                current_frame_ids.add(track.track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                track_history[track.track_id].append((cx, cy))
                if len(track_history[track.track_id]) > 30:
                    track_history[track.track_id].pop(0)

        current_in_zone = current_frame_ids

        cv2.putText(frame, f'In Zone: {len(current_in_zone)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Click 5 points for zone", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, "Press 'r' to reset zone", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        for track_id, points in track_history.items():
            for i in range(1, len(points)):
                cv2.line(frame, points[i-1], points[i], (255, 0, 0), 2)

        cv2.imshow('Zone People Counter', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            zone_points.clear()
            zone_ready[0] = False

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    zone_counter()
