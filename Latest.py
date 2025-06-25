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
    print("COMPLETE PEOPLE COUNTING SYSTEM - YOLOv8m Optimized")
    print("Detection + Tracking + AREA/ZONE Counting")
    print("=" * 60)

def print_statistics_header():
    """Print statistics section header"""
    print("\n" + "=" * 60)
    print("PROCESSING STATISTICS")
    print("=" * 60)

class CountingZone:
    """Polygon zone for counting people entering/exiting areas"""
    def __init__(self, points, zone_id="main_zone"):
        """
        Initialize counting zone
        points: list of (x, y) tuples defining polygon vertices
        zone_id: unique identifier for this zone
        """
        self.points = np.array(points, dtype=np.int32)
        self.zone_id = zone_id
        
        # Tracking for zone crossings
        self.track_positions = {}  # Store previous positions of tracked objects
        self.zone_events = []      # Store all zone crossing events
        
        # Calculate zone properties
        self.calculate_zone_properties()
        
    def calculate_zone_properties(self):
        """Calculate zone center and bounding box for display"""
        self.center_x = int(np.mean(self.points[:, 0]))
        self.center_y = int(np.mean(self.points[:, 1]))
        self.center = (self.center_x, self.center_y)
        
        # Bounding box for optimization
        self.min_x = np.min(self.points[:, 0])
        self.max_x = np.max(self.points[:, 0])
        self.min_y = np.min(self.points[:, 1])
        self.max_y = np.max(self.points[:, 1])
    
    def point_in_zone(self, point):
        """Check if a point is inside the polygon zone using OpenCV"""
        x, y = point
        
        # Quick bounding box check first (optimization)
        if x < self.min_x or x > self.max_x or y < self.min_y or y > self.max_y:
            return False
        
        # Precise polygon check
        result = cv2.pointPolygonTest(self.points, (float(x), float(y)), False)
        return result >= 0  # >= 0 means inside or on boundary
    
    def check_zone_crossing(self, track_id, current_position):
        """Check if a tracked object entered or exited the zone"""
        current_in_zone = self.point_in_zone(current_position)
        
        if track_id not in self.track_positions:
            # First time seeing this track
            self.track_positions[track_id] = {
                'position': current_position,
                'in_zone': current_in_zone,
                'frame_count': 1,
                'stable_frames': 0
            }
            return None
        
        prev_data = self.track_positions[track_id]
        prev_in_zone = prev_data['in_zone']
        
        # Update position data
        self.track_positions[track_id] = {
            'position': current_position,
            'in_zone': current_in_zone,
            'frame_count': prev_data['frame_count'] + 1,
            'stable_frames': prev_data['stable_frames'] + 1 if current_in_zone == prev_in_zone else 0
        }
        
        # Check for zone crossing (status change)
        # Only count if the person has been stable for a few frames to avoid noise
        min_stable_frames = 3  # Minimum frames to confirm crossing
        
        if (prev_in_zone != current_in_zone and 
            prev_data['stable_frames'] >= min_stable_frames):
            
            # Determine direction
            if not prev_in_zone and current_in_zone:
                direction = "ENTERED"
            elif prev_in_zone and not current_in_zone:
                direction = "EXITED"
            else:
                return None  # Should not happen
            
            crossing_event = {
                'track_id': track_id,
                'direction': direction,
                'timestamp': time.time(),
                'position': current_position,
                'zone_id': self.zone_id
            }
            
            self.zone_events.append(crossing_event)
            return crossing_event
        
        return None
    
    def draw_zone(self, frame):
        """Draw the counting zone on frame"""
        # Draw zone polygon (semi-transparent)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.points], (0, 255, 255))  # Yellow fill
        cv2.addWeighted(frame, 0.8, overlay, 0.2, 0, frame)
        
        # Draw zone border (thick)
        cv2.polylines(frame, [self.points], True, (0, 255, 255), 3)
        
        # Add zone label
        cv2.putText(frame, f"COUNTING ZONE: {self.zone_id}", 
                   (self.center_x - 80, self.center_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame

# COMMENTED OUT: Line counting implementation
"""
class CountingLine:
    # Virtual line for counting people crossing
    def __init__(self, start_point, end_point, line_id="main"):
        self.start_point = start_point  # (x1, y1)
        self.end_point = end_point      # (x2, y2)
        self.line_id = line_id
        
        # Calculate line properties
        self.calculate_line_properties()
        
        # Crossing detection
        self.previous_positions = {}  # Store previous positions of tracked objects
        self.crossings = []          # Store all crossing events
        
    def calculate_line_properties(self):
        # Calculate mathematical properties of the line
        x1, y1 = self.start_point
        x2, y2 = self.end_point
        
        # Line equation: Ax + By + C = 0
        self.A = y2 - y1
        self.B = x1 - x2  
        self.C = x2 * y1 - x1 * y2
        
        # Line length
        self.length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def point_side(self, point):
        # Determine which side of line a point is on
        x, y = point
        value = self.A * x + self.B * y + self.C
        
        if value > 0:
            return 1    # One side
        elif value < 0:
            return -1   # Other side
        else:
            return 0    # Exactly on line
    
    def check_crossing(self, track_id, current_position):
        # Check if a tracked object crossed the line
        if track_id not in self.previous_positions:
            # First time seeing this track
            self.previous_positions[track_id] = {
                'position': current_position,
                'side': self.point_side(current_position),
                'frame_count': 1
            }
            return None
        
        prev_data = self.previous_positions[track_id]
        prev_side = prev_data['side']
        current_side = self.point_side(current_position)
        
        # Update position
        self.previous_positions[track_id] = {
            'position': current_position,
            'side': current_side,
            'frame_count': prev_data['frame_count'] + 1
        }
        
        # Check for crossing (side change)
        if prev_side != 0 and current_side != 0 and prev_side != current_side:
            # Determine direction
            direction = "IN" if (prev_side == -1 and current_side == 1) else "OUT"
            
            crossing_event = {
                'track_id': track_id,
                'direction': direction,
                'timestamp': time.time(),
                'position': current_position,
                'line_id': self.line_id
            }
            
            self.crossings.append(crossing_event)
            return crossing_event
        
        return None
    
    def draw_line(self, frame):
        # Draw the counting line on frame
        # Draw main line (thick blue)
        cv2.line(frame, self.start_point, self.end_point, (255, 0, 0), 3)
        
        # Draw direction indicators
        # Calculate perpendicular points for arrows
        mid_x = (self.start_point[0] + self.end_point[0]) // 2
        mid_y = (self.start_point[1] + self.end_point[1]) // 2
        
        # Add labels
        cv2.putText(frame, "COUNTING LINE", (mid_x - 50, mid_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return frame
"""

class SimpleTracker:
    """
    Simplified tracking using centroid distance matching
    (Same as previous implementations)
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
    """Complete people counting system with area/zone counting"""
    
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
        
        # Area/Zone counting system (replacing line counting)
        self.counting_zones = []
        self.total_counts = {
            'ENTERED': 0,
            'EXITED': 0,
            'CURRENT_IN_ZONE': 0,  # People currently inside the zone
            'NET': 0
        }
        self.zone_events_history = []
        
        # Track statistics
        self.track_stats = {
            'total_tracks_created': 0,
            'active_tracks': 0,
            'max_simultaneous_tracks': 0
        }
        
    def add_counting_zone(self, points, zone_id="main_zone"):
        """Add a polygon counting zone"""
        zone = CountingZone(points, zone_id)
        self.counting_zones.append(zone)
        print(f"Counting zone added: {len(points)} points, ID: {zone_id}")
        return zone
    
    def setup_default_counting_zone(self, frame_width, frame_height):
        """Setup a default rectangular counting zone in the center of frame"""
        # Create a rectangular zone in the center (can be modified for your needs)
        margin_x = frame_width // 4
        margin_y = frame_height // 4
        
        zone_points = [
            (margin_x, margin_y),                           # Top-left
            (frame_width - margin_x, margin_y),             # Top-right
            (frame_width - margin_x, frame_height - margin_y), # Bottom-right
            (margin_x, frame_height - margin_y)              # Bottom-left
        ]
        
        return self.add_counting_zone(zone_points, "default_center_zone")
    
    def update_current_occupancy(self):
        """Update count of people currently in zones"""
        current_in_zone = 0
        
        for zone in self.counting_zones:
            for track_id, track_data in zone.track_positions.items():
                if track_data['in_zone']:
                    current_in_zone += 1
        
        self.total_counts['CURRENT_IN_ZONE'] = current_in_zone
        self.total_counts['NET'] = self.total_counts['ENTERED'] - self.total_counts['EXITED']
    
    def detect_and_track(self, frame):
        """Detect persons and update tracking with zone counting"""
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
        
        # Stage 3: Zone Counting (replacing line counting)
        new_zone_events = []
        for zone in self.counting_zones:
            for object_id, (centroid, box) in tracked_objects.items():
                zone_event = zone.check_zone_crossing(object_id, centroid)
                if zone_event:
                    new_zone_events.append(zone_event)
                    
                    # Update total counts
                    direction = zone_event['direction']
                    self.total_counts[direction] += 1
                    
                    # Store in history
                    self.zone_events_history.append(zone_event)
                    
                    print(f"[xD] ZONE EVENT: Person #{object_id} {direction} zone '{zone_event['zone_id']}' | "
                          f"Total: ENTERED={self.total_counts['ENTERED']}, EXITED={self.total_counts['EXITED']}")
        
        # Update current occupancy
        self.update_current_occupancy()
        
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
        annotated_frame = self.draw_complete_visualization(frame, tracked_objects, new_zone_events)
        
        return len(detections), tracked_objects, new_zone_events, annotated_frame
    
    def draw_complete_visualization(self, frame, tracked_objects, new_zone_events):
        """Draw complete visualization with tracking + zone counting"""
        annotated_frame = frame.copy()
        
        # Draw counting zones
        for zone in self.counting_zones:
            annotated_frame = zone.draw_zone(annotated_frame)
        
        # Draw tracked objects
        for object_id, (centroid, box) in tracked_objects.items():
            x1, y1, x2, y2 = map(int, box)
            
            # Check if this person just had a zone event
            just_crossed = any(e['track_id'] == object_id for e in new_zone_events)
            
            # Check if person is currently in any zone
            in_zone = False
            for zone in self.counting_zones:
                if object_id in zone.track_positions and zone.track_positions[object_id]['in_zone']:
                    in_zone = True
                    break
            
            # Color coding: Yellow if just crossed, Green if in zone, Blue if outside
            if just_crossed:
                color = (0, 255, 255)  # Yellow - just crossed
            elif in_zone:
                color = (0, 255, 0)    # Green - currently in zone
            else:
                color = (255, 0, 0)    # Blue - outside zone
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw centroid
            cv2.circle(annotated_frame, centroid, 5, color, -1)
            
            # Add ID label with status
            status = ""
            if just_crossed:
                zone_event = next(e for e in new_zone_events if e['track_id'] == object_id)
                status = f" ({zone_event['direction']})"
            elif in_zone:
                status = " (IN)"
            
            label = f"ID: {object_id}{status}"
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw counting statistics
        self.draw_count_display(annotated_frame)
        
        return annotated_frame
    
    def draw_count_display(self, frame):
        """Draw zone counting statistics on frame"""
        # Background for count display
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Count text
        cv2.putText(frame, "ZONE COUNT", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"ENTERED:  {self.total_counts['ENTERED']}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(frame, f"EXITED:   {self.total_counts['EXITED']}", (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.putText(frame, f"IN ZONE:  {self.total_counts['CURRENT_IN_ZONE']}", (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.putText(frame, f"NET:      {self.total_counts['NET']}", (20, 135), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Active tracks
        cv2.putText(frame, f"Active: {self.track_stats['active_tracks']}", (200, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Total IDs: {self.track_stats['total_tracks_created']}", (200, 85), 
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
            'zone_counting_stats': self.total_counts,
            'total_zone_events': len(self.zone_events_history)
        }

def process_video_with_zone_counting(video_path, show_video=True, save_video=False):
    """Process video with complete people zone counting system"""
    
    print_header()
    
    # Initialize counting system
    counting_system = PeopleCountingSystem()
    
    # Open video
    print(f"Processing video: {video_path}")
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
    
    # Setup counting zone (rectangular area in center - you can customize this)
    counting_system.setup_default_counting_zone(width, height)
    print("Note: Using default center zone. You can customize zone points in the code.")
    
    # Setup video writer if saving
    out = None
    if save_video:
        output_path = video_path.replace('.mp4', '_zone_counted.mp4')
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
        
        # Complete processing: Detection + Tracking + Zone Counting
        detection_count, tracked_objects, zone_events, annotated_frame = counting_system.detect_and_track(frame)
        
        # Add frame info
        active_tracks = len(tracked_objects)
        info_text = (f"Frame: {frame_num}/{total_frames} | "
                    f"Detections: {detection_count} | "
                    f"Tracks: {active_tracks} | "
                    f"ENTERED: {counting_system.total_counts['ENTERED']} | "
                    f"EXITED: {counting_system.total_counts['EXITED']} | "
                    f"IN ZONE: {counting_system.total_counts['CURRENT_IN_ZONE']}")
        
        cv2.putText(annotated_frame, info_text, (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Save frame if requested
        if out:
            out.write(annotated_frame)
        
        # Show video if requested
        if show_video:
            cv2.imshow('Zone-Based People Counting System', annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nUser requested quit")
                break
            elif key == ord(' '):
                cv2.waitKey(0)  # Pause on spacebar
        
        # Show progress updates
        if frame_num % progress_interval == 0 or frame_num == total_frames:
            progress = (frame_num / total_frames) * 100
            print(f"Progress: {progress:.1f}% - "
                  f"{detection_count} detections, {active_tracks} tracks, "
                  f"ENTERED: {counting_system.total_counts['ENTERED']}, "
                  f"EXITED: {counting_system.total_counts['EXITED']}, "
                  f"IN ZONE: {counting_system.total_counts['CURRENT_IN_ZONE']}")
    
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
        print(f"Average detections per frame: {stats['avg_detections']:.1f}")
        print(f"Total processing time: {stats['total_processing_time']:.1f}s")
        print(f"Total unique tracks created: {stats['tracking_stats']['total_tracks_created']}")
        print(f"Max simultaneous tracks: {stats['tracking_stats']['max_simultaneous_tracks']}")
        print(f"")
        print(f"ZONE COUNTING RESULTS:")
        print(f"People entered zone: {stats['zone_counting_stats']['ENTERED']}")
        print(f"People exited zone: {stats['zone_counting_stats']['EXITED']}")
        print(f"Currently in zone: {stats['zone_counting_stats']['CURRENT_IN_ZONE']}")
        print(f"Net count: {stats['zone_counting_stats']['NET']}")
        print(f"Total zone events detected: {stats['total_zone_events']}")
        print("=" * 60)
    
    return stats

# Main execution
if __name__ == "__main__":
    # Configuration
    VIDEO_PATH = "503653056_9712668558855145_1521156624328842562_n.mp4"
    SHOW_VIDEO = True  
    SAVE_VIDEO = False  
    
    # Process the video
    try:
        statistics = process_video_with_zone_counting(VIDEO_PATH, show_video=SHOW_VIDEO, save_video=SAVE_VIDEO)
        
        if statistics:
            print(f"\n[SUCCESS] Complete zone-based people counting system working!")
            print(f"Final counts - ENTERED: {statistics['zone_counting_stats']['ENTERED']}, "
                  f"EXITED: {statistics['zone_counting_stats']['EXITED']}, "
                  f"CURRENTLY IN ZONE: {statistics['zone_counting_stats']['CURRENT_IN_ZONE']}, "
                  f"NET: {statistics['zone_counting_stats']['NET']}")
        else:
            print("[ERROR] Processing failed!")
            
    except FileNotFoundError:
        print(f"[ERROR] Video file '{VIDEO_PATH}' not found!")
        print("Please check the file path and try again.")
    except Exception as e:
        print(f"[ERROR] {e}")

# USAGE NOTES:
"""
ZONE/AREA COUNTING FEATURES:

1. POLYGON ZONE DEFINITION:
   - Define any polygon shape for counting area
   - Default: rectangular zone in center of frame
   - Customize by modifying setup_default_counting_zone() or calling add_counting_zone()

2. COUNTING LOGIC:
   - ENTERED: Person enters the defined zone
   - EXITED: Person exits the defined zone  
   - CURRENT_IN_ZONE: People currently inside the zone
   - NET: Total entered minus total exited

3. ANTI-NOISE FEATURES:
   - Minimum stable frames (3) before counting crossing
   - Polygon-based precise zone detection
   - Track consistency validation

4. VISUAL FEEDBACK:
   - Yellow semi-transparent zone overlay
   - Color-coded people: Yellow (just crossed), Green (in zone), Blue (outside)
   - Real-time count display

5. CUSTOMIZATION OPTIONS:
   To customize the counting zone, modify the points in setup_default_counting_zone():
   
   # Example: Entrance door area
   zone_points = [
       (300, 200),  # Top-left
       (500, 200),  # Top-right
       (500, 400),  # Bottom-right
       (300, 400)   # Bottom-left
   ]
   
   # Example: Complex polygon (L-shaped area)
   zone_points = [
       (100, 100),
       (300, 100),
       (300, 200),
       (200, 200),
       (200, 300),
       (100, 300)
   ]

ADVANTAGES OF ZONE COUNTING VS LINE COUNTING:
- Better for complex entry/exit patterns
- Handles people stopping in doorways
- More accurate for wide entrances
- Can count occupancy (people currently inside)
- Handles bidirectional movement naturally
"""