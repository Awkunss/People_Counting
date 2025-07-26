#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zone Counting Module  
Module đếm người trong vùng khu vực
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import defaultdict
import sys
import os

# Add paths for imports
current_dir = os.path.dirname(__file__)
tracking_dir = os.path.join(current_dir, 'tracking')
config_dir = os.path.join(current_dir, '..', '..', 'config')

sys.path.insert(0, tracking_dir)
sys.path.insert(0, config_dir)

from deep_tracker import DeepTracker
from settings import CLASS_ID, CONF_THRESHOLD, get_model_path, get_video_path

# Global variables
zone_points = []
zone_ready = False
current_in_zone = set()
track_history = defaultdict(list)
crossing_history = {}
people_entering = 0
people_leaving = 0

def draw_zone(frame):
    """Vẽ vùng zone - chỉ đường viền"""
    global zone_points, zone_ready
    
    if zone_ready and len(zone_points) > 2:
        # Vẽ đường viền polygon màu xanh lá
        pts = np.array(zone_points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Vẽ các điểm góc với số thứ tự
        for i, point in enumerate(zone_points):
            cv2.circle(frame, point, 8, (0, 255, 0), -1)
            cv2.putText(frame, str(i+1), (point[0]+10, point[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

def draw_trajectories(frame):
    """Vẽ trajectories của các track"""
    for track_id, points in track_history.items():
        if len(points) > 1:
            for i in range(1, len(points)):
                cv2.line(frame, points[i-1], points[i], (255, 0, 255), 2)

def draw_stats(frame):
    """Vẽ thống kê trên frame"""
    global current_in_zone, people_entering, people_leaving
    
    # Background box
    cv2.rectangle(frame, (10, 10), (300, 150), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (300, 150), (255, 255, 255), 2)
    
    # Text
    cv2.putText(frame, "ZONE COUNTING", (20, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"In Zone: {len(current_in_zone)}", (20, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Entering: {people_entering}", (20, 85), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Leaving: {people_leaving}", (20, 110), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def reset_system():
    """Reset counting system"""
    global current_in_zone, track_history, crossing_history, people_entering, people_leaving, zone_points, zone_ready
    
    current_in_zone = set()
    track_history = defaultdict(list)
    crossing_history = {}
    people_entering = 0
    people_leaving = 0
    zone_points = []
    zone_ready = False
    print("System reset!")

def mouse_callback(event, x, y, flags, param):
    """Mouse callback cho vẽ zone"""
    global zone_points, zone_ready
    
    if event == cv2.EVENT_LBUTTONDOWN:
        zone_points.append((x, y))
        print(f"Point {len(zone_points)}: ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(zone_points) >= 3:
            zone_ready = True
            print(f"Zone set with {len(zone_points)} points")
        else:
            print("Need at least 3 points for zone!")

def zone_counter(video_path='Test.mp4', model='yolov8s.pt'):
    """Main zone counting function"""
    global zone_points, zone_ready, current_in_zone, track_history, crossing_history
    global people_entering, people_leaving
    
    # Sử dụng config paths
    full_video_path = get_video_path(video_path)
    full_model_path = get_model_path(model)
    
    # Khởi tạo
    if isinstance(full_video_path, int):
        cap = cv2.VideoCapture(full_video_path)
    else:
        cap = cv2.VideoCapture(str(full_video_path))
    
    model = YOLO(str(full_model_path))
    tracker = DeepTracker(max_age=30, min_hits=3)
    
    cv2.namedWindow('People Counting - Zone')
    cv2.setMouseCallback('People Counting - Zone', mouse_callback)
    
    print("🏢 ZONE COUNTING")
    print("📝 Click các điểm để vẽ zone, chuột phải để hoàn thành")
    print("⚠️ ESC để thoát, R để reset")
    
    fps_counter = 0
    fps_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if isinstance(full_video_path, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            
            fps_counter += 1
            
            # Vẽ zone
            draw_zone(frame)
            
            # YOLO detection
            results = model(frame, classes=[CLASS_ID], conf=CONF_THRESHOLD, verbose=False)
            detections = []
            
            if results and results[0].boxes.xyxy is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box[:4])
                    detections.append([x1, y1, x2, y2])
            
            # Tracking và zone counting
            active_tracks = tracker.update(detections, frame)
            
            # Đếm người trong zone
            current_frame_ids = set()
            for track in active_tracks:
                x1, y1, x2, y2 = map(int, track.bbox)
                cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)
                
                # Kiểm tra người có trong zone không
                in_zone = False
                if zone_ready and len(zone_points) > 2:
                    pts = np.array(zone_points, np.int32).reshape((-1, 1, 2))
                    in_zone = cv2.pointPolygonTest(pts, (cx, cy), False) >= 0
                
                # Zone crossing detection
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
                
                # Update crossing history
                crossing_history[track_id] = in_zone
                
                if in_zone:
                    current_frame_ids.add(track.track_id)
                    # Vẽ người trong zone với màu đỏ
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"ID:{track.track_id}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Lưu trajectory
                    track_history[track.track_id].append((cx, cy))
                    if len(track_history[track.track_id]) > 30:
                        track_history[track.track_id].pop(0)
                else:
                    # Vẽ người ngoài zone với màu xanh dương
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"ID:{track.track_id}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Clean up old crossing history
            active_track_ids = {track.track_id for track in active_tracks}
            crossing_history = {tid: status for tid, status in crossing_history.items() 
                               if tid in active_track_ids}
            
            # Cập nhật current_in_zone
            current_in_zone = current_frame_ids
            
            # Vẽ trajectories và stats
            draw_trajectories(frame)
            draw_stats(frame)
            
            # FPS display
            if fps_counter % 30 == 0:
                current_time = time.time()
                fps = 30 / (current_time - fps_time) if current_time > fps_time else 0
                fps_time = current_time
                print(f"FPS: {fps:.1f} | In Zone: {len(current_in_zone)} | Entering: {people_entering} | Leaving: {people_leaving}")
            
            cv2.imshow('People Counting - Zone', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('r') or key == ord('R'):
                reset_system()
    
    except KeyboardInterrupt:
        print("\nDừng bởi user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"📊 Kết quả cuối: In Zone: {len(current_in_zone)}, Entering: {people_entering}, Leaving: {people_leaving}")

if __name__ == "__main__":
    zone_counter()
