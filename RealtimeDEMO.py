import cv2
from ultralytics import YOLO
import time
import numpy as np
from collections import defaultdict
import math
import torch

def print_header():
    """Print professional header"""
    print("=" * 60)
    print("COMPLETE PEOPLE COUNTING SYSTEM - YOLOv8 Line Counting")
    print("Detection + Tracking + LINE Crossing Counting")
    print("=" * 60)

def print_statistics_header():
    """Print statistics section header"""
    print("\n" + "=" * 60)
    print("PROCESSING STATISTICS")
    print("=" * 60)

class CountingLine:
    """Virtual line for counting people crossing"""
    def __init__(self, start_point, end_point, line_id="main"):
        """
        Initialize counting line
        start_point: (x1, y1) tuple - start of line
        end_point: (x2, y2) tuple - end of line
        line_id: unique identifier for this line
        """
        self.start_point = start_point  # (x1, y1)
        self.end_point = end_point      # (x2, y2)
        self.line_id = line_id
        
        # Calculate line properties
        self.calculate_line_properties()
        
        # Crossing detection
        self.previous_positions = {}  # Store previous positions of tracked objects
        self.crossings = []          # Store all crossing events
        
    def calculate_line_properties(self):
        """Calculate mathematical properties of the line"""
        x1, y1 = self.start_point
        x2, y2 = self.end_point
        
        # Line equation: Ax + By + C = 0
        self.A = y2 - y1
        self.B = x1 - x2  
        self.C = x2 * y1 - x1 * y2
        
        # Line length and center
        self.length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        self.center_x = (x1 + x2) // 2
        self.center_y = (y1 + y2) // 2
        self.center = (self.center_x, self.center_y)
    
    def point_side(self, point):
        """Determine which side of line a point is on"""
        x, y = point
        value = self.A * x + self.B * y + self.C
        
        if value > 0:
            return 1    # One side
        elif value < 0:
            return -1   # Other side
        else:
            return 0    # Exactly on line
    
    def check_crossing(self, track_id, current_position):
        """Check if a tracked object crossed the line"""
        current_side = self.point_side(current_position)
        
        if track_id not in self.previous_positions:
            # First time seeing this track
            self.previous_positions[track_id] = {
                'position': current_position,
                'side': current_side,
                'frame_count': 1,
                'stable_frames': 0
            }
            return None
        
        prev_data = self.previous_positions[track_id]
        prev_side = prev_data['side']
        
        # Update position data
        self.previous_positions[track_id] = {
            'position': current_position,
            'side': current_side,
            'frame_count': prev_data['frame_count'] + 1,
            'stable_frames': prev_data['stable_frames'] + 1 if current_side == prev_side else 0
        }
        
        # Check for crossing (side change) - SIMPLIFIED LOGIC
        # Only need to check if sides are different and both are non-zero
        if (prev_side != 0 and current_side != 0 and prev_side != current_side):
            
            # CORRECTED direction detection
            # For horizontal lines: 
            # - Going from top to bottom (above line to below line) = IN
            # - Going from bottom to top (below line to above line) = OUT
            
            if abs(self.A) <= abs(self.B):  # More horizontal line
                direction = "IN" if (prev_side == 1 and current_side == -1) else "OUT"
            else:  # More vertical line
                direction = "IN" if (prev_side == -1 and current_side == 1) else "OUT"
            
            crossing_event = {
                'track_id': track_id,
                'direction': direction,
                'timestamp': time.time(),
                'position': current_position,
                'line_id': self.line_id,
                'prev_side': prev_side,
                'current_side': current_side
            }
            
            self.crossings.append(crossing_event)
            print(f"[DEBUG] Track {track_id}: {prev_side} -> {current_side} = {direction}")
            return crossing_event
        
        return None
    
    def draw_line(self, frame):
        """Draw the counting line on frame"""
        # Draw main line (thicker and more visible)
        cv2.line(frame, self.start_point, self.end_point, (0, 0, 255), 6)
        
        # Draw direction indicators (arrows)
        # Calculate perpendicular direction for arrows
        dx = self.end_point[0] - self.start_point[0]
        dy = self.end_point[1] - self.start_point[1]
        
        # Normalize direction
        length = math.sqrt(dx*dx + dy*dy)
        if length > 0:
            dx = dx / length
            dy = dy / length
            
            # Perpendicular direction
            perp_dx = -dy
            perp_dy = dx
            
            # Arrow points
            arrow_length = 40
            mid_x, mid_y = self.center
            
            # Draw direction arrows (IN side - GREEN)
            arrow_end_x = int(mid_x + perp_dx * arrow_length)
            arrow_end_y = int(mid_y + perp_dy * arrow_length)
            cv2.arrowedLine(frame, (mid_x, mid_y), (arrow_end_x, arrow_end_y), 
                           (0, 255, 0), 4, tipLength=0.3)
            cv2.putText(frame, "IN", (arrow_end_x + 10, arrow_end_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Draw direction arrows (OUT side - RED)
            arrow_end_x = int(mid_x - perp_dx * arrow_length)
            arrow_end_y = int(mid_y - perp_dy * arrow_length)
            cv2.arrowedLine(frame, (mid_x, mid_y), (arrow_end_x, arrow_end_y), 
                           (0, 0, 255), 4, tipLength=0.3)
            cv2.putText(frame, "OUT", (arrow_end_x + 10, arrow_end_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Add line label (larger and more visible)
        cv2.putText(frame, f"COUNTING LINE: {self.line_id}", 
                   (self.center_x - 100, self.center_y - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
        
        # Add line equation info for debugging
        cv2.putText(frame, f"Line Eq: {self.A:.1f}x + {self.B:.1f}y + {self.C:.1f} = 0", 
                   (self.center_x - 120, self.center_y + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Add larger circles at endpoints
        cv2.circle(frame, self.start_point, 12, (0, 0, 255), -1)
        cv2.circle(frame, self.end_point, 12, (0, 0, 255), -1)
        
        return frame

class SimpleTracker:
    """
    Simplified tracking using centroid distance matching
    """
    def __init__(self, max_disappeared=30, max_distance=100):
        self.next_id = 0
        self.objects = {}  
        self.disappeared = {}  
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
    
    def register(self, centroid):
        """Register a new object"""
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1
    
    def deregister(self, object_id):
        """Remove an object from tracking"""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def calculate_centroid(self, box):
        """Calculate centroid from bounding box [x1, y1, x2, y2]"""
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        return (cx, cy)
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def update(self, detections):
        """Update tracker with new detections"""
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return {}
        
        input_centroids = []
        for detection in detections:
            centroid = self.calculate_centroid(detection)
            input_centroids.append(centroid)
        
        if len(self.objects) == 0:
            for i, centroid in enumerate(input_centroids):
                self.register(centroid)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            distance_matrix = np.zeros((len(object_centroids), len(input_centroids)))
            for i in range(len(object_centroids)):
                for j in range(len(input_centroids)):
                    distance_matrix[i][j] = self.calculate_distance(
                        object_centroids[i], input_centroids[j]
                    )
            
            rows = distance_matrix.min(axis=1).argsort()
            cols = distance_matrix.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                if distance_matrix[row, col] <= self.max_distance:
                    object_id = object_ids[row]
                    self.objects[object_id] = input_centroids[col]
                    self.disappeared[object_id] = 0
                    
                    used_rows.add(row)
                    used_cols.add(col)
            
            unused_rows = set(range(0, distance_matrix.shape[0])).difference(used_rows)
            unused_cols = set(range(0, distance_matrix.shape[1])).difference(used_cols)
            
            if distance_matrix.shape[0] >= distance_matrix.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])
        
        result = {}
        for i, (object_id, centroid) in enumerate(self.objects.items()):
            min_dist = float('inf')
            best_box = None
            for detection in detections:
                det_centroid = self.calculate_centroid(detection)
                dist = self.calculate_distance(centroid, det_centroid)
                if dist < min_dist:
                    min_dist = dist
                    best_box = detection
            
            if best_box is not None and min_dist <= self.max_distance:
                result[object_id] = (centroid, best_box)
        
        return result

class PeopleCountingSystem:
    """Complete people counting system with line crossing counting"""
    
    def __init__(self, model_name='yolov8s.pt', confidence_threshold=0.25):
        self.confidence_threshold = confidence_threshold
        self.frame_count = 0
        self.total_inference_time = 0
        self.total_detections = 0
        self.start_time = None
        
        # GPU acceleration setup
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_half_precision = self.device == 'cuda'  # FP16 for GPU
        
        # Initialize detector
        print(f"Loading {model_name} model...")
        self.model = YOLO(model_name)
        
        # Move model to GPU if available
        if self.device == 'cuda':
            self.model.to(self.device)
            print(f"[SUCCESS] Model loaded on GPU: {torch.cuda.get_device_name()}")
            print(f"[INFO] Using FP16 precision for optimal performance")
        else:
            print(f"[WARNING] GPU not available, using CPU")
            
        print(f"Device: {self.device}")
        
        # Initialize tracker
        self.tracker = SimpleTracker(max_disappeared=30, max_distance=100)
        print("Tracker initialized: Simple Centroid Tracker")
        
        # Line counting system
        self.counting_lines = []
        self.total_counts = {
            'IN': 0,
            'OUT': 0,
            'NET': 0
        }
        self.crossing_events_history = []
        
        # Track statistics
        self.track_stats = {
            'total_tracks_created': 0,
            'active_tracks': 0,
            'max_simultaneous_tracks': 0
        }
        
    def add_counting_line(self, start_point, end_point, line_id="main"):
        """Add a counting line"""
        line = CountingLine(start_point, end_point, line_id)
        self.counting_lines.append(line)
        print(f"Counting line added: {start_point} to {end_point}, ID: {line_id}")
        return line
    
    def setup_default_counting_line(self, frame_width, frame_height):
        """Setup a default horizontal counting line in the frame"""
        # Create a horizontal line across the frame (slightly below upper third)
        margin = frame_width // 6  # Leave some margin on sides
        line_y = int(frame_height * 0.7)  # About 40% down from top (lowered a bit)
        
        start_point = (margin, line_y)
        end_point = (frame_width - margin, line_y)
        
        return self.add_counting_line(start_point, end_point, "default_horizontal_line")
    
    def detect_and_track(self, frame):
        """Detect persons and update tracking with line counting"""
        start_time = time.time()
        
        # Stage 1: Detection with GPU acceleration
        results = self.model(
            frame, 
            classes=[0], 
            conf=self.confidence_threshold, 
            device=self.device,
            half=self.use_half_precision,  # FP16 for GPU speed
            verbose=False
        )
        
        inference_time = time.time() - start_time
        self.total_inference_time += inference_time
        self.frame_count += 1
        
        # Extract detections
        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            for box in boxes:
                detections.append(box)
        
        # Stage 2: Tracking
        tracked_objects = self.tracker.update(detections)
        
        # Stage 3: Line Crossing Detection
        new_crossing_events = []
        for line in self.counting_lines:
            for object_id, (centroid, box) in tracked_objects.items():
                crossing_event = line.check_crossing(object_id, centroid)
                if crossing_event:
                    new_crossing_events.append(crossing_event)
                    
                    # Update total counts
                    direction = crossing_event['direction']
                    self.total_counts[direction] += 1
                    self.total_counts['NET'] = self.total_counts['IN'] - self.total_counts['OUT']
                    
                    # Store in history
                    self.crossing_events_history.append(crossing_event)
                    
                    print(f"[CROSSING] Person #{object_id} crossed {direction} on line '{crossing_event['line_id']}' | "
                          f"Total: IN={self.total_counts['IN']}, OUT={self.total_counts['OUT']}, NET={self.total_counts['NET']}")
        
        # Update statistics
        current_track_count = len(tracked_objects)
        self.track_stats['active_tracks'] = current_track_count
        self.track_stats['max_simultaneous_tracks'] = max(
            self.track_stats['max_simultaneous_tracks'], 
            current_track_count
        )
        
        if hasattr(self.tracker, 'next_id'):
            self.track_stats['total_tracks_created'] = self.tracker.next_id
        
        self.total_detections += len(detections)
        
        # Create annotated frame
        annotated_frame = self.draw_complete_visualization(frame, tracked_objects, new_crossing_events)
        
        return len(detections), tracked_objects, new_crossing_events, annotated_frame
    
    def draw_complete_visualization(self, frame, tracked_objects, new_crossing_events):
        """Draw complete visualization with tracking + line counting"""
        annotated_frame = frame.copy()
        
        # Draw counting lines
        for line in self.counting_lines:
            annotated_frame = line.draw_line(annotated_frame)
        
        # Draw tracked objects
        for object_id, (centroid, box) in tracked_objects.items():
            x1, y1, x2, y2 = map(int, box)
            
            # Check if this person just crossed a line
            just_crossed = any(e['track_id'] == object_id for e in new_crossing_events)
            
            # Color coding: Yellow if just crossed, Blue otherwise
            if just_crossed:
                color = (0, 255, 255)  # Yellow - just crossed
                crossing_event = next(e for e in new_crossing_events if e['track_id'] == object_id)
                status = f" ({crossing_event['direction']})"
            else:
                color = (255, 0, 0)    # Blue - normal tracking
                status = ""
                
                # Show which side of line person is on (DEBUG INFO)
                for line in self.counting_lines:
                    if object_id in line.previous_positions:
                        side = line.previous_positions[object_id]['side']
                        status = f" (Side:{side})"
                        break
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw centroid (larger for better visibility)
            cv2.circle(annotated_frame, centroid, 8, color, -1)
            
            # Add ID label with status
            label = f"ID: {object_id}{status}"
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw trajectory line (optional - shows movement)
            for line in self.counting_lines:
                if object_id in line.previous_positions:
                    prev_pos = line.previous_positions[object_id]['position']
                    cv2.line(annotated_frame, prev_pos, centroid, (128, 128, 128), 2)
        
        # Draw counting statistics
        self.draw_count_display(annotated_frame)
        
        return annotated_frame
    
    def draw_count_display(self, frame):
        """Draw line crossing statistics on frame"""
        # Background for count display
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (280, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Count text
        cv2.putText(frame, "LINE CROSSINGS", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"IN:  {self.total_counts['IN']}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(frame, f"OUT: {self.total_counts['OUT']}", (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.putText(frame, f"NET: {self.total_counts['NET']}", (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Active tracks
        cv2.putText(frame, f"Tracks: {self.track_stats['active_tracks']}", (150, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Total: {self.track_stats['total_tracks_created']}", (150, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def get_statistics(self):
        """Calculate and return performance statistics"""
        if self.frame_count == 0:
            return None
            
        avg_inference_time = self.total_inference_time / self.frame_count
        avg_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        avg_detections = self.total_detections / self.frame_count
        
        return {
            'total_frames': self.frame_count,
            'avg_inference_time': avg_inference_time,
            'avg_fps': avg_fps,
            'avg_detections': avg_detections,
            'total_processing_time': time.time() - self.start_time if self.start_time else 0,
            'tracking_stats': self.track_stats,
            'line_counting_stats': self.total_counts,
            'total_crossing_events': len(self.crossing_events_history)
        }

def process_video_with_line_counting(video_path, show_video=True, save_video=False):
    """Process video file with complete people line counting system (kept for backward compatibility)"""
    
    print_header()
    print(f"[VIDEO MODE] Processing: {video_path}")
    
    # Initialize counting system
    counting_system = PeopleCountingSystem()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup counting line
    counting_system.setup_default_counting_line(width, height)
    
    # Setup video writer if saving
    out = None
    if save_video:
        output_path = video_path.replace('.mp4', '_line_counted.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Output will be saved to: {output_path}")
    
    # Start processing
    counting_system.start_time = time.time()
    frame_num = 0
    progress_interval = max(1, total_frames // 10)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        
        # Complete processing
        detection_count, tracked_objects, crossing_events, annotated_frame = counting_system.detect_and_track(frame)
        
        # Add frame info
        info_text = (f"Frame: {frame_num}/{total_frames} | "
                    f"IN: {counting_system.total_counts['IN']} | "
                    f"OUT: {counting_system.total_counts['OUT']} | "
                    f"NET: {counting_system.total_counts['NET']}")
        
        cv2.putText(annotated_frame, info_text, (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        if out:
            out.write(annotated_frame)
        
        if show_video:
            cv2.imshow('Video Line Counting', annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # Progress updates
        if frame_num % progress_interval == 0:
            progress = (frame_num / total_frames) * 100
            print(f"Progress: {progress:.1f}% - IN: {counting_system.total_counts['IN']}, OUT: {counting_system.total_counts['OUT']}")
    
    # Cleanup
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    return counting_system.get_statistics()

def process_camera_with_line_counting(camera_id=1, show_video=True, save_video=False, camera_width=1280, camera_height=720):
    """Process real-time camera feed with complete people line counting system"""
    
    print_header()
    
    # Initialize counting system
    counting_system = PeopleCountingSystem()
    
    # Open camera
    print(f"Opening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        print("Available options:")
        print("- Check if camera is connected")
        print("- Try different camera_id (0, 1, 2, etc.)")
        print("- Check camera permissions")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    cap.set(cv2.CAP_PROP_FPS, 60)  # Try to set 30 FPS
    
    # Get actual camera properties
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera opened successfully!")
    print(f"Resolution: {actual_width}x{actual_height}")
    print(f"FPS: {actual_fps:.1f}")
    
    # Setup counting line
    counting_system.setup_default_counting_line(actual_width, actual_height)
    print("Counting line positioned. Press 'q' to quit, 'r' to reset counts, 's' to save frame")
    
    # Setup video writer if saving
    out = None
    if save_video:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"camera_counting_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (actual_width, actual_height))
        print(f"Recording to: {output_path}")
    
    # Start processing
    counting_system.start_time = time.time()
    frame_count = 0
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0
    
    print("\n🚀 REAL-TIME COUNTING STARTED")
    print("=" * 50)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame from camera")
            break
        
        frame_count += 1
        fps_frame_count += 1
        
        # Calculate real-time FPS every second
        if time.time() - fps_start_time >= 1.0:
            current_fps = fps_frame_count / (time.time() - fps_start_time)
            fps_frame_count = 0
            fps_start_time = time.time()
        
        # Complete processing: Detection + Tracking + Line Counting
        detection_count, tracked_objects, crossing_events, annotated_frame = counting_system.detect_and_track(frame)
        
        # Add real-time info overlay
        active_tracks = len(tracked_objects)
        
        # Real-time stats overlay
        info_text = (f"LIVE | FPS: {current_fps:.1f} | "
                    f"Detections: {detection_count} | "
                    f"Tracks: {active_tracks} | "
                    f"IN: {counting_system.total_counts['IN']} | "
                    f"OUT: {counting_system.total_counts['OUT']} | "
                    f"NET: {counting_system.total_counts['NET']}")
        
        cv2.putText(annotated_frame, info_text, (10, actual_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add timestamp
        timestamp_text = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated_frame, timestamp_text, (10, actual_height - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Print crossing events in real-time
        for event in crossing_events:
            print(f"[{timestamp_text}] Person #{event['track_id']} {event['direction']} | "
                  f"Total: IN={counting_system.total_counts['IN']}, "
                  f"OUT={counting_system.total_counts['OUT']}, "
                  f"NET={counting_system.total_counts['NET']}")
        
        # Save frame if requested
        if out:
            out.write(annotated_frame)
        
        # Show video
        if show_video:
            cv2.imshow('Real-Time People Counting System', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('r'):
                # Reset counts
                counting_system.total_counts = {'IN': 0, 'OUT': 0, 'NET': 0}
                counting_system.crossing_events_history = []
                for line in counting_system.counting_lines:
                    line.crossings = []
                    line.previous_positions = {}
                print("\n[RESET] Counts reset to zero")
            elif key == ord('s'):
                # Save current frame
                save_filename = f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(save_filename, annotated_frame)
                print(f"\n[SAVED] Frame saved as {save_filename}")
            elif key == ord(' '):
                # Pause/unpause
                print("\n[PAUSED] Press any key to continue...")
                cv2.waitKey(0)
                print("[RESUMED] Continuing...")
        
        # Print stats every 100 frames
        if frame_count % 100 == 0:
            runtime = time.time() - counting_system.start_time
            print(f"\n[STATS] Runtime: {runtime:.0f}s | Frames: {frame_count} | "
                  f"Avg FPS: {frame_count/runtime:.1f} | "
                  f"Total Crossings: {len(counting_system.crossing_events_history)}")
    
    # Cleanup
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    stats = counting_system.get_statistics()
    if stats:
        print_statistics_header()
        print(f"Total frames processed: {stats['total_frames']}")
        print(f"Average inference time: {stats['avg_inference_time']:.3f}s")
        print(f"Average FPS: {stats['avg_fps']:.1f}")
        print(f"Total runtime: {stats['total_processing_time']:.1f}s")
        print(f"Total unique tracks created: {stats['tracking_stats']['total_tracks_created']}")
        print(f"Max simultaneous tracks: {stats['tracking_stats']['max_simultaneous_tracks']}")
        print(f"")
        print(f"FINAL LINE CROSSING RESULTS:")
        print(f"People crossed IN: {stats['line_counting_stats']['IN']}")
        print(f"People crossed OUT: {stats['line_counting_stats']['OUT']}")
        print(f"Net count: {stats['line_counting_stats']['NET']}")
        print(f"Total crossing events: {stats['total_crossing_events']}")
        print("=" * 60)
    
    return stats

# Main execution
if __name__ == "__main__":
    
    # CHOOSE MODE: "camera" or "video"
    MODE = "camera"  # Change to "video" to process video files
    
    if MODE == "camera":
        # Real-time camera configuration
        CAMERA_ID = 1      # 0 = default camera, 1 = second camera, etc.
        SHOW_VIDEO = True      # Show live feed
        SAVE_VIDEO = False     # Record to file
        CAMERA_WIDTH = 1280    # Camera resolution width
        CAMERA_HEIGHT = 720    # Camera resolution height
        
        print("🎥 REAL-TIME PEOPLE COUNTING SYSTEM")
        print("=" * 50)
        print("Controls:")
        print("  'q' - Quit")
        print("  'r' - Reset counts")
        print("  's' - Save current frame")
        print("  ' ' - Pause/Resume")
        print("=" * 50)
        
        # Process the camera feed
        try:
            statistics = process_camera_with_line_counting(
                camera_id=CAMERA_ID,
                show_video=SHOW_VIDEO, 
                save_video=SAVE_VIDEO,
                camera_width=CAMERA_WIDTH,
                camera_height=CAMERA_HEIGHT
            )
            
            if statistics:
                print(f"\n[SUCCESS] Real-time counting system completed!")
                print(f"Final counts - IN: {statistics['line_counting_stats']['IN']}, "
                      f"OUT: {statistics['line_counting_stats']['OUT']}, "
                      f"NET: {statistics['line_counting_stats']['NET']}")
            else:
                print("[ERROR] Camera processing failed!")
                
        except KeyboardInterrupt:
            print(f"\n[STOPPED] System stopped by user (Ctrl+C)")
            print(f"[ERROR] {e}")
            print("\nTroubleshooting:")
            print("1. Check if camera is connected and working")
            print("2. Try different CAMERA_ID (0, 1, 2, etc.)")
            print("3. Check camera permissions")
            print("4. Ensure no other application is using the camera")
    
    elif MODE == "video":
        # Video file configuration
        VIDEO_PATH = "503653056_9712668558855145_1521156624328842562_n.mp4"
        SHOW_VIDEO = True  
        SAVE_VIDEO = False  
        
        print("📹 VIDEO FILE PEOPLE COUNTING SYSTEM")
        print("=" * 50)
        
        # Process the video
        try:
            statistics = process_video_with_line_counting(VIDEO_PATH, show_video=SHOW_VIDEO, save_video=SAVE_VIDEO)
            
            if statistics:
                print(f"\n[SUCCESS] Video processing completed!")
                print(f"Final counts - IN: {statistics['line_counting_stats']['IN']}, "
                      f"OUT: {statistics['line_counting_stats']['OUT']}, "
                      f"NET: {statistics['line_counting_stats']['NET']}")
            else:
                print("[ERROR] Video processing failed!")
                
        except FileNotFoundError:
            print(f"[ERROR] Video file '{VIDEO_PATH}' not found!")
            print("Please check the file path and try again.")
        except Exception as e:
            print(f"[ERROR] {e}")
    
    else:
        print("[ERROR] Invalid MODE. Set MODE = 'camera' or MODE = 'video'")

# REAL-TIME CAMERA vs VIDEO MODES:
"""
🎥 DUAL MODE SYSTEM: CAMERA + VIDEO

CAMERA MODE (Real-time):
- Live camera feed processing
- Interactive controls (reset, save, pause)
- Real-time FPS monitoring
- Instant crossing notifications
- Perfect for: Live monitoring, demonstrations, real-time applications

VIDEO MODE (File processing):
- Process pre-recorded video files
- Batch processing with progress tracking
- Good for: Analysis, testing, processing recorded footage

QUICK SETUP:

1. For CAMERA mode:
   MODE = "camera"
   CAMERA_ID = 0        # Your camera
   
2. For VIDEO mode:
   MODE = "video" 
   VIDEO_PATH = "your_video.mp4"

CAMERA FEATURES:
✅ Real-time processing
✅ Interactive controls ('q', 'r', 's', ' ')
✅ Live FPS counter
✅ Instant notifications
✅ Optional recording
✅ Auto-timestamping
✅ Snapshot saving

VIDEO FEATURES:  
✅ File processing
✅ Progress tracking
✅ Batch analysis
✅ Frame-by-frame control
✅ Output video generation

CONTROLS (Camera mode):
- 'q' = Quit
- 'r' = Reset counts to zero
- 's' = Save current frame
- ' ' = Pause/Resume
- Ctrl+C = Emergency stop

TROUBLESHOOTING:
Camera not working? Try:
1. Different CAMERA_ID (0, 1, 2...)
2. Check camera permissions
3. Close other camera apps
4. Check USB connection
"""