import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import torch

CLASS_ID = 0  # ID lớp 'person'
CONF_THRESHOLD = 0.5

def draw_zone(event, x, y, flags, param):
    zone_points, zone_ready = param
    if event == cv2.EVENT_LBUTTONDOWN and not zone_ready[0]:
        zone_points.append((x, y))
        print(f"Point {len(zone_points)}: ({x}, {y})")
        if len(zone_points) == 5:
            zone_ready[0] = True
            print("Zone is ready!")

def zone_people_counter(
    video_path='C:/Users/ACER/Downloads/DAT/Test.mp4',
    model_path='yolov9s.pt'
):
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

    model = YOLO(model_path).to('cuda:0')
    zone_points = []
    zone_ready = [False]  # Dùng list để truyền tham chiếu
    current_in_zone = set()
    track_history = defaultdict(list)

    cap = cv2.VideoCapture(video_path)
    cv2.namedWindow('Zone People Counter')
    cv2.setMouseCallback('Zone People Counter', draw_zone, param=(zone_points, zone_ready))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Vẽ các điểm đã chọn
        for i, point in enumerate(zone_points):
            cv2.circle(frame, point, 8, (0, 255, 0), -1)
            cv2.putText(frame, str(i+1), (point[0]+10, point[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Vẽ đa giác khi đã đủ 5 điểm
        if zone_ready[0]:
            pts = np.array(zone_points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        else:
            pts = None

        # Nhận diện và theo dõi đối tượng
        results = model.track(
            source=frame,
            classes=[CLASS_ID],
            conf=CONF_THRESHOLD
        )

        # Kiểm tra đối tượng trong vùng
        current_frame_ids = set()
        if results[0].boxes.id is not None and zone_ready[0]:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                center = (int(x), int(y))
                # Kiểm tra điểm tâm có nằm trong vùng zone
                if cv2.pointPolygonTest(pts, center, False) >= 0:
                    current_frame_ids.add(track_id)
                    cv2.rectangle(frame,
                                  (int(x - w/2), int(y - h/2)),
                                  (int(x + w/2), int(y + h/2)),
                                  (0, 0, 255), 2)
                    track_history[track_id].append(center)
                    if len(track_history[track_id]) > 30:
                        track_history[track_id].pop(0)
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
    zone_people_counter()
