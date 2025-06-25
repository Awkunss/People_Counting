import cv2
from ultralytics import YOLO
import time
import numpy as np
from collections import defaultdict
import math

def print_header():
    """Print professional header"""
    print("=" * 60)
    print("PERSON DETECTION + TRACKING SYSTEM - Stage 2 Implementation")
    print("=" * 60)

def print_statistics_header():
    """Print statistics section header"""
    print("\n" + "=" * 60)
    print("PROCESSING STATISTICS")
    print("=" * 60)

class SimpleTracker:
    """
    Simplified tracking using centroid distance matching
    Good for edge devices with limited computational resources
    """
    def __init__(self, max_disappeared=30, max_distance=100):
        self.next_id = 0
        self.objects = {}  # Current tracked objects
        self.disappeared = {}  # Count frames since object disappeared
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
        """
        Update tracker with new detections
        detections: list of bounding boxes [[x1, y1, x2, y2], ...]
        Returns: dict {object_id: (centroid, box)}
        """
        # If no detections, mark all existing objects as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return {}
        
        # Calculate centroids for all detections
        input_centroids = []
        for detection in detections:
            centroid = self.calculate_centroid(detection)
            input_centroids.append(centroid)
        
        # If no existing objects, register all detections as new objects
        if len(self.objects) == 0:
            for i, centroid in enumerate(input_centroids):
                self.register(centroid)
        else:
            # Get existing object centroids and IDs
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Compute distance matrix between existing objects and new detections
            distance_matrix = np.zeros((len(object_centroids), len(input_centroids)))
            for i in range(len(object_centroids)):
                for j in range(len(input_centroids)):
                    distance_matrix[i][j] = self.calculate_distance(
                        object_centroids[i], input_centroids[j]
                    )
            
            # Find the minimum distance assignments
            rows = distance_matrix.min(axis=1).argsort()
            cols = distance_matrix.argmin(axis=1)[rows]
            
            # Keep track of used row and column indices
            used_rows = set()
            used_cols = set()
            
            # Update existing objects with closest detections
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                if distance_matrix[row, col] <= self.max_distance:
                    object_id = object_ids[row]
                    self.objects[object_id] = input_centroids[col]
                    self.disappeared[object_id] = 0
                    
                    used_rows.add(row)
                    used_cols.add(col)
            
            # Handle unmatched detections and objects
            unused_rows = set(range(0, distance_matrix.shape[0])).difference(used_rows)
            unused_cols = set(range(0, distance_matrix.shape[1])).difference(used_cols)
            
            # If more objects than detections, mark unmatched objects as disappeared
            if distance_matrix.shape[0] >= distance_matrix.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            
            # If more detections than objects, register new objects
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])
        
        # Return current objects with their bounding boxes
        result = {}
        for i, (object_id, centroid) in enumerate(self.objects.items()):
            # Find the closest detection box for this centroid
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

class PersonDetectorTracker:
    def __init__(self, model_name='yolov8m-crowdhuman.pt', confidence_threshold=0.25):
        self.confidence_threshold = confidence_threshold
        self.frame_count = 0
        self.total_inference_time = 0
        self.total_detections = 0
        self.start_time = None
        
        # Initialize tracker
        self.tracker = SimpleTracker(max_disappeared=30, max_distance=100)
        
        # Track statistics
        self.track_stats = {
            'total_tracks_created': 0,
            'active_tracks': 0,
            'max_simultaneous_tracks': 0
        }
        
        print(f"Loading {model_name} model...")
        self.model = YOLO(model_name)
        print(f"Model loaded on device: {self.model.device}")
        print("Tracker initialized: Simple Centroid Tracker")
    
    def detect_and_track(self, frame):
        """Detect persons and update tracking"""
        start_time = time.time()
        
        # Run YOLO inference - only detect persons (class 0)
        results = self.model(frame, classes=[0], conf=self.confidence_threshold, verbose=False)
        
        inference_time = time.time() - start_time
        self.total_inference_time += inference_time
        self.frame_count += 1
        
        # Extract detections
        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confidences):
                detections.append(box)
        
        # Update tracker
        tracked_objects = self.tracker.update(detections)
        
        # Update statistics
        current_track_count = len(tracked_objects)
        self.track_stats['active_tracks'] = current_track_count
        self.track_stats['max_simultaneous_tracks'] = max(
            self.track_stats['max_simultaneous_tracks'], 
            current_track_count
        )
        
        # Keep track of total unique tracks created
        if hasattr(self.tracker, 'next_id'):
            self.track_stats['total_tracks_created'] = self.tracker.next_id
        
        self.total_detections += len(detections)
        
        # Create annotated frame
        annotated_frame = self.draw_tracks(frame, tracked_objects, detections)
        
        return len(detections), tracked_objects, annotated_frame
    
    def draw_tracks(self, frame, tracked_objects, raw_detections):
        """Draw tracking results on frame"""
        annotated_frame = frame.copy()
        
        # Draw tracked objects with IDs
        for object_id, (centroid, box) in tracked_objects.items():
            x1, y1, x2, y2 = map(int, box)
            
            # Draw bounding box (green for tracked objects)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw centroid
            cv2.circle(annotated_frame, centroid, 5, (0, 255, 0), -1)
            
            # Add ID label
            label = f"ID: {object_id}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                        (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return annotated_frame
    
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
            'tracking_stats': self.track_stats
        }

def process_video_with_tracking(video_path, show_video=True, save_video=False):
    """Process video with detection and tracking"""
    
    print_header()
    
    # Initialize detector and tracker
    detector_tracker = PersonDetectorTracker()
    
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
    
    # Setup video writer if saving
    out = None
    if save_video:
        output_path = video_path.replace('.mp4', '_tracked.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Output will be saved to: {output_path}")
    
    # Start processing
    detector_tracker.start_time = time.time()
    frame_num = 0
    progress_interval = max(1, total_frames // 10)  # Show progress 10 times
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        
        # Detect and track persons
        detection_count, tracked_objects, annotated_frame = detector_tracker.detect_and_track(frame)
        
        # Add frame info to display
        active_tracks = len(tracked_objects)
        info_text = f"Frame: {frame_num}/{total_frames} | Detections: {detection_count} | Active Tracks: {active_tracks}"
        cv2.putText(annotated_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save frame if requested
        if out:
            out.write(annotated_frame)
        
        # Show video if requested
        if show_video:
            cv2.imshow('Person Detection + Tracking - Stage 2', annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nUser requested quit")
                break
            elif key == ord(' '):
                cv2.waitKey(0)  # Pause on spacebar
        
        # Show progress updates
        if frame_num % progress_interval == 0 or frame_num == total_frames:
            progress = (frame_num / total_frames) * 100
            print(f"Progress: {progress:.1f}% - {detection_count} detections, {active_tracks} active tracks")
    
    # Cleanup
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    stats = detector_tracker.get_statistics()
    if stats:
        print_statistics_header()
        print(f"Total frames processed: {stats['total_frames']}")
        print(f"Average inference time: {stats['avg_inference_time']:.3f}s")
        print(f"Average FPS: {stats['avg_fps']:.1f}")
        print(f"Average detections per frame: {stats['avg_detections']:.1f}")
        print(f"Total processing time: {stats['total_processing_time']:.1f}s")
        print(f"Total unique tracks created: {stats['tracking_stats']['total_tracks_created']}")
        print(f"Max simultaneous tracks: {stats['tracking_stats']['max_simultaneous_tracks']}")
        print("=" * 60)
    
    return stats

# Main execution
if __name__ == "__main__":
    # Configuration
    VIDEO_PATH = "test23.mp4"  # Change this to your video file
    SHOW_VIDEO = True  # Set to False if you don't want to see the video
    SAVE_VIDEO = False  # Set to True if you want to save the output video
    
    # Process the video
    try:
        statistics = process_video_with_tracking(VIDEO_PATH, show_video=SHOW_VIDEO, save_video=SAVE_VIDEO)
        
        if statistics:
            print(f"\n^_^ Stage 2 completed successfully!")
            print(f"Processed {statistics['total_frames']} frames at {statistics['avg_fps']:.1f} FPS")
            print(f"Created {statistics['tracking_stats']['total_tracks_created']} unique tracks")
        else:
            print("!!! Processing failed!")
            
    except FileNotFoundError:
        print(f"!!! Error: Video file '{VIDEO_PATH}' not found!")
        print("Please check the file path and try again.")
    except Exception as e:
        print(f"!!! Error: {e}")