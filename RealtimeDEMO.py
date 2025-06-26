import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import torch
import math
import time
import json
import argparse
import os

# Configuration
CLASS_ID = 0  # ID for 'person' class
CONF_THRESHOLD = 0.5

# Global variables for line counting
line_points = []
line_ready = [False]
total_counts = {'IN': 0, 'OUT': 0, 'NET': 0}
previous_positions = {}
crossing_events = []
track_history = defaultdict(list)

# Line configuration presets
LINE_PRESETS = {
    'horizontal_center': lambda w, h: [(50, h//2), (w-50, h//2)],
    'vertical_center': lambda w, h: [(w//2, 50), (w//2, h-50)],
    'diagonal_tl_br': lambda w, h: [(w//4, h//4), (3*w//4, 3*h//4)],
    'diagonal_tr_bl': lambda w, h: [(3*w//4, h//4), (w//4, 3*h//4)],
    'entrance_door': lambda w, h: [(w//3, h//3), (2*w//3, h//3)],
    'exit_door': lambda w, h: [(w//4, 3*h//4), (3*w//4, 3*h//4)]
}

def load_config_file(config_path="line_config.json"):
    """Load line configuration from JSON file"""
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config.get('counting_lines', {})
        else:
            # Create default config file
            default_config = {
                "counting_lines": {
                    "main_entrance": [[200, 300], [600, 300]],
                    "side_door": [[100, 500], [400, 500]],
                    "hallway": [[320, 100], [320, 600]]
                }
            }
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            print(f"Created default config file: {config_path}")
            return default_config['counting_lines']
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}

def setup_line_from_config(frame, line_config):
    """Setup counting line from various configuration options"""
    global line_points, line_ready
    
    h, w = frame.shape[:2]
    
    if line_config is None:
        return False
    
    # Case 1: Direct coordinates [(x1,y1), (x2,y2)]
    if isinstance(line_config, list) and len(line_config) == 2:
        if all(isinstance(p, (list, tuple)) and len(p) == 2 for p in line_config):
            line_points = [tuple(line_config[0]), tuple(line_config[1])]
            line_ready[0] = True
            calculate_line_properties()
            print(f"Line set from coordinates: {line_points}")
            return True
    
    # Case 2: Preset name
    if isinstance(line_config, str) and line_config in LINE_PRESETS:
        line_points = LINE_PRESETS[line_config](w, h)
        line_ready[0] = True
        calculate_line_properties()
        print(f"Line set from preset '{line_config}': {line_points}")
        return True
    
    # Case 3: From config file
    if isinstance(line_config, str):
        config_lines = load_config_file()
        if line_config in config_lines:
            line_points = [tuple(config_lines[line_config][0]), tuple(config_lines[line_config][1])]
            line_ready[0] = True
            calculate_line_properties()
            print(f"Line set from config file '{line_config}': {line_points}")
            return True
    
    return False

def draw_line(event, x, y, flags, param):
    """Mouse callback function to draw counting line"""
    global line_points, line_ready
    
    if event == cv2.EVENT_LBUTTONDOWN and not line_ready[0]:
        line_points.append((x, y))
        print(f"Point {len(line_points)}: ({x}, {y})")
        if len(line_points) == 2:
            line_ready[0] = True
            print("Counting line is ready!")
            # Calculate line properties
            calculate_line_properties()

def calculate_line_properties():
    """Calculate mathematical properties of the counting line"""
    global line_A, line_B, line_C, line_center
    
    if len(line_points) < 2:
        return
        
    x1, y1 = line_points[0]
    x2, y2 = line_points[1]
    
    # Line equation: Ax + By + C = 0
    line_A = y2 - y1
    line_B = x1 - x2  
    line_C = x2 * y1 - x1 * y2
    
    # Line center
    line_center = ((x1 + x2) // 2, (y1 + y2) // 2)

def point_side(point):
    """Determine which side of line a point is on"""
    x, y = point
    value = line_A * x + line_B * y + line_C
    
    if value > 0:
        return 1    # One side
    elif value < 0:
        return -1   # Other side
    else:
        return 0    # Exactly on line

def check_crossing(track_id, current_position):
    """Check if a tracked object crossed the line"""
    global previous_positions, total_counts, crossing_events
    
    current_side = point_side(current_position)
    
    if track_id not in previous_positions:
        # First time seeing this track
        previous_positions[track_id] = {
            'position': current_position,
            'side': current_side,
            'frame_count': 1
        }
        return None
    
    prev_data = previous_positions[track_id]
    prev_side = prev_data['side']
    
    # Update position data
    previous_positions[track_id] = {
        'position': current_position,
        'side': current_side,
        'frame_count': prev_data['frame_count'] + 1
    }
    
    # Check for crossing (side change)
    if (prev_side != 0 and current_side != 0 and prev_side != current_side):
        
        # Direction detection
        if abs(line_A) <= abs(line_B):  # More horizontal line
            direction = "IN" if (prev_side == 1 and current_side == -1) else "OUT"
        else:  # More vertical line
            direction = "IN" if (prev_side == -1 and current_side == 1) else "OUT"
        
        crossing_event = {
            'track_id': track_id,
            'direction': direction,
            'timestamp': time.time(),
            'position': current_position,
            'prev_side': prev_side,
            'current_side': current_side
        }
        
        crossing_events.append(crossing_event)
        
        # Update total counts
        total_counts[direction] += 1
        total_counts['NET'] = total_counts['IN'] - total_counts['OUT']
        
        print(f"[CROSSING] Person #{track_id}: {direction} | "
              f"Total: IN={total_counts['IN']}, OUT={total_counts['OUT']}, NET={total_counts['NET']}")
        
        return crossing_event
    
    return None

def draw_counting_line(frame):
    """Draw the counting line on frame"""
    if len(line_points) < 2:
        return frame
    
    # Draw main line
    cv2.line(frame, line_points[0], line_points[1], (0, 0, 255), 6)
    
    # Draw direction indicators
    if line_ready[0]:
        dx = line_points[1][0] - line_points[0][0]
        dy = line_points[1][1] - line_points[0][1]
        
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
            mid_x, mid_y = line_center
            
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
    
    # Draw endpoints
    for i, point in enumerate(line_points):
        cv2.circle(frame, point, 12, (0, 0, 255), -1)
        cv2.putText(frame, str(i+1), (point[0]+15, point[1]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add line label
    if line_ready[0]:
        cv2.putText(frame, "COUNTING LINE", 
                   (line_center[0] - 80, line_center[1] - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return frame

def draw_statistics(frame):
    """Draw counting statistics on frame"""
    # Background for count display
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (280, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Count text
    cv2.putText(frame, "LINE CROSSINGS", (20, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(frame, f"IN:  {total_counts['IN']}", (20, 65), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.putText(frame, f"OUT: {total_counts['OUT']}", (20, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.putText(frame, f"NET: {total_counts['NET']}", (20, 115), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Active tracks
    active_tracks = len([t for t in previous_positions.keys()])
    cv2.putText(frame, f"Tracks: {active_tracks}", (150, 65), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def reset_counting_system():
    """Reset all counting variables"""
    global line_points, line_ready, total_counts, previous_positions, crossing_events, track_history
    
    line_points.clear()
    line_ready[0] = False
    total_counts = {'IN': 0, 'OUT': 0, 'NET': 0}
    previous_positions.clear()
    crossing_events.clear()
    track_history.clear()
    print("System reset!")

def show_available_configs():
    """Show all available line configurations"""
    print("\n" + "="*50)
    print("AVAILABLE LINE CONFIGURATIONS")
    print("="*50)
    
    print("\n1. PRESET LINES:")
    for preset in LINE_PRESETS.keys():
        print(f"   - {preset}")
    
    print("\n2. CONFIG FILE LINES:")
    config_lines = load_config_file()
    for name in config_lines.keys():
        print(f"   - {name}: {config_lines[name]}")
    
    print("\n3. MANUAL COORDINATES:")
    print("   - Format: [[x1,y1], [x2,y2]]")
    print("   - Example: [[100,200], [400,200]]")
    
    print("\n4. COMMAND LINE ARGS:")
    print("   - --line x1 y1 x2 y2")
    print("   - --preset preset_name") 
    print("   - --config config_name")
    print("="*50)

def line_people_counter(
    video_path='C:/Users/ACER/Downloads/DAT/Test.mp4',
    model_path='yolov8s.pt',
    camera_id=None,
    line_config=None,
    show_config_help=False
):
    """
    Main function for line crossing people counter
    
    Args:
        video_path: Path to video file (if camera_id is None)
        model_path: Path to YOLO model
        camera_id: Camera ID for live feed (0, 1, etc.) - overrides video_path
        line_config: Line configuration (coordinates, preset name, or config name)
        show_config_help: Show available configurations and exit
    """
    global line_points, line_ready, total_counts, previous_positions
    
    if show_config_help:
        show_available_configs()
        return
    
    print("=" * 60)
    print("FUNCTIONAL LINE CROSSING PEOPLE COUNTER")
    print("Detection + Tracking + Line Crossing Counting")
    print("=" * 60)
    
    # System info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    if torch.cuda.is_available():
        model = model.to('cuda:0')
    
    # Open video source
    if camera_id is not None:
        print(f"Opening camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        source_name = f"Camera {camera_id}"
    else:
        print(f"Opening video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        source_name = video_path
    
    if not cap.isOpened():
        print(f"Error: Could not open {source_name}")
        return
    
    # Get first frame to setup line
    success, first_frame = cap.read()
    if not success:
        print("Error: Could not read first frame")
        return
    
    # Setup counting line from config
    line_configured = False
    if line_config:
        line_configured = setup_line_from_config(first_frame, line_config)
    
    # Set up window and mouse callback
    window_name = 'Line Crossing People Counter'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_line)
    
    print("\nInstructions:")
    if line_configured:
        print("✓ Counting line configured automatically")
    else:
        print("1. Click 2 points to draw counting line")
    print("2. Press 'r' to reset line and counts")
    print("3. Press 'q' to quit")
    print("4. Press 's' to save current frame")
    print("5. Press 'h' to show config help")
    print("=" * 60)
    
    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            if camera_id is not None:
                print("Failed to grab frame from camera")
                break
            else:
                print("End of video file")
                break
        
        frame_count += 1
        original_frame = frame.copy()
        
        # Draw counting line
        frame = draw_counting_line(frame)
        
        # Run detection and tracking
        results = model.track(
            source=frame,
            classes=[CLASS_ID],
            conf=CONF_THRESHOLD,
            verbose=False
        )
        
        # Process detections for line crossing
        new_crossings = []
        if results[0].boxes.id is not None and line_ready[0]:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for box, track_id, conf in zip(boxes, track_ids, confidences):
                x, y, w, h = box
                center = (int(x), int(y))
                
                # Check for line crossing
                crossing_event = check_crossing(track_id, center)
                if crossing_event:
                    new_crossings.append(crossing_event)
                
                # Update track history
                track_history[track_id].append(center)
                if len(track_history[track_id]) > 30:
                    track_history[track_id].pop(0)
                
                # Draw bounding box
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                
                # Color coding: Yellow if just crossed, Blue otherwise
                just_crossed = any(e['track_id'] == track_id for e in new_crossings)
                color = (0, 255, 255) if just_crossed else (255, 0, 0)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, center, 6, color, -1)
                
                # Add track ID and confidence
                label = f"ID: {track_id} ({conf:.2f})"
                if just_crossed:
                    crossing = next(e for e in new_crossings if e['track_id'] == track_id)
                    label += f" {crossing['direction']}"
                
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw track history (trajectories)
        for track_id, points in track_history.items():
            for i in range(1, len(points)):
                cv2.line(frame, points[i-1], points[i], (128, 128, 128), 2)
        
        # Draw statistics
        draw_statistics(frame)
        
        # Add instructions
        if not line_ready[0]:
            cv2.putText(frame, "Click 2 points to draw counting line", (10, frame.shape[0] - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.putText(frame, "Press 'r' to reset, 'q' to quit, 's' to save, 'h' for help", (10, frame.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Add frame info
        runtime = time.time() - start_time
        fps = frame_count / runtime if runtime > 0 else 0
        cv2.putText(frame, f"Frame: {frame_count} | FPS: {fps:.1f}", (10, frame.shape[0] - 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow(window_name, frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            reset_counting_system()
        elif key == ord('s'):
            # Save current frame
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"line_counter_snapshot_{timestamp}.jpg"
            cv2.imwrite(filename, original_frame)
            print(f"Frame saved as: {filename}")
        elif key == ord('h'):
            show_available_configs()
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    runtime = time.time() - start_time
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    print(f"Total frames processed: {frame_count}")
    print(f"Total runtime: {runtime:.1f}s")
    print(f"Average FPS: {fps:.1f}")
    print(f"People crossed IN: {total_counts['IN']}")
    print(f"People crossed OUT: {total_counts['OUT']}")
    print(f"Net count: {total_counts['NET']}")
    print(f"Total crossing events: {len(crossing_events)}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Line Crossing People Counter")
    parser.add_argument('--video', type=str, default='C:/Users/ACER/Downloads/DAT/Test.mp4',
                       help='Path to video file')
    parser.add_argument('--camera', type=int, help='Camera ID (overrides video)')
    parser.add_argument('--model', type=str, default='yolov8s.pt',
                       help='Path to YOLO model')
    parser.add_argument('--line', nargs=4, type=int, metavar=('x1', 'y1', 'x2', 'y2'),
                       help='Line coordinates: x1 y1 x2 y2')
    parser.add_argument('--preset', type=str, choices=list(LINE_PRESETS.keys()),
                       help='Use predefined line preset')
    parser.add_argument('--config', type=str,
                       help='Use line from config file')
    parser.add_argument('--show-configs', action='store_true',
                       help='Show available configurations and exit')
    
    args = parser.parse_args()
    
    # Determine line configuration
    line_config = None
    if args.line:
        line_config = [(args.line[0], args.line[1]), (args.line[2], args.line[3])]
    elif args.preset:
        line_config = args.preset
    elif args.config:
        line_config = args.config
    
    # Run the counter
    line_people_counter(
        video_path=args.video,
        model_path=args.model,
        camera_id=args.camera,
        line_config=line_config,
        show_config_help=args.show_configs
    )