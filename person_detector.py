import cv2
from ultralytics import YOLO
import time
import numpy as np

def print_header():
    """Print professional header"""
    print("=" * 60)
    print("PERSON DETECTION SYSTEM - Stage 1 Implementation")
    print("=" * 60)

def print_statistics_header():
    """Print statistics section header"""
    print("\n" + "=" * 60)
    print("PROCESSING STATISTICS")
    print("=" * 60)

class PersonDetector:
    def __init__(self, model_name='yolov8n.pt', confidence_threshold=0.25):
        self.confidence_threshold = confidence_threshold
        self.frame_count = 0
        self.total_inference_time = 0
        self.total_detections = 0
        self.start_time = None
        
        print(f"Loading {model_name} model...")
        self.model = YOLO(model_name)
        print(f"Model loaded on device: {self.model.device}")
    
    def detect_persons(self, frame):
        """Detect persons in frame and return count"""
        start_time = time.time()
        
        # Run YOLO inference - only detect persons (class 0)
        results = self.model(frame, classes=[0], conf=self.confidence_threshold, verbose=False)
        
        inference_time = time.time() - start_time
        self.total_inference_time += inference_time
        self.frame_count += 1
        
        # Count detections
        person_count = 0
        annotated_frame = frame.copy()
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            person_count = len(boxes)
            self.total_detections += person_count
            
            # Draw bounding boxes
            for box, conf in zip(boxes, confidences):
                x1, y1, x2, y2 = map(int, box)
                
                # Draw rectangle
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
                label = f"Person: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return person_count, annotated_frame
    
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
            'total_processing_time': time.time() - self.start_time if self.start_time else 0
        }

def process_video(video_path, show_video=True, save_video=False):
    """Process video with professional output formatting"""
    
    print_header()
    
    # Initialize detector
    detector = PersonDetector()
    
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
        output_path = video_path.replace('.mp4', '_detected.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Output will be saved to: {output_path}")
    
    # Start processing
    detector.start_time = time.time()
    frame_num = 0
    progress_interval = max(1, total_frames // 10)  # Show progress 10 times
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        
        # Detect persons
        person_count, annotated_frame = detector.detect_persons(frame)
        
        # Add frame info to display
        info_text = f"Frame: {frame_num}/{total_frames} | Persons: {person_count}"
        cv2.putText(annotated_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save frame if requested
        if out:
            out.write(annotated_frame)
        
        # Show video if requested
        if show_video:
            cv2.imshow('Person Detection - Stage 1', annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nUser requested quit")
                break
            elif key == ord(' '):
                cv2.waitKey(0)  # Pause on spacebar
        
        # Show progress updates
        if frame_num % progress_interval == 0 or frame_num == total_frames:
            progress = (frame_num / total_frames) * 100
            print(f"Progress: {progress:.1f}% - {person_count} persons detected in current frame")
    
    # Cleanup
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    stats = detector.get_statistics()
    if stats:
        print_statistics_header()
        print(f"Total frames processed: {stats['total_frames']}")
        print(f"Average inference time: {stats['avg_inference_time']:.3f}s")
        print(f"Average FPS: {stats['avg_fps']:.1f}")
        print(f"Average detections per frame: {stats['avg_detections']:.1f}")
        print(f"Total processing time: {stats['total_processing_time']:.1f}s")
        print("=" * 60)
    
    return stats

# Main execution
if __name__ == "__main__":
    # Configuration
    VIDEO_PATH = "test_1.mp4"  # Change this to your video file
    SHOW_VIDEO = True  # Set to False if you don't want to see the video
    SAVE_VIDEO = False  # Set to True if you want to save the output video
    
    # Process the video
    try:
        statistics = process_video(VIDEO_PATH, show_video=SHOW_VIDEO, save_video=SAVE_VIDEO)
        
        if statistics:
            print(f"\n^_^ Processing completed successfully!")
            print(f"Processed {statistics['total_frames']} frames at {statistics['avg_fps']:.1f} FPS")
        else:
            print("!!! Processing failed!")
            
    except FileNotFoundError:
        print(f"!!! Error: Video file '{VIDEO_PATH}' not found!")
        print("Please check the file path and try again.")
    except Exception as e:
        print(f"!!! Error: {e}")