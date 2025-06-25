import cv2
from ultralytics import YOLO
import time
import numpy as np
from collections import defaultdict
import math
import torch
import psutil
import json
import yaml
from datetime import datetime
from pathlib import Path
import gc
import os

def print_header():
    """Print professional header"""
    print("=" * 80)
    print("CUSTOM DATASET BENCHMARK - YOUR PEOPLE COUNTING DATASET")
    print("YOLO Models Evaluation on DAT.v1i.yolov8 Dataset")
    print("=" * 80)

def print_statistics_header():
    """Print statistics section header"""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS ON YOUR CUSTOM DATASET")
    print("=" * 80)

class ModelConfig:
    """Configuration for each model to test"""
    def __init__(self, name, model_file, category, expected_params, notes=""):
        self.name = name
        self.model_file = model_file
        self.category = category
        self.expected_params = expected_params
        self.notes = notes

# Models to benchmark - Focus on core YOLO versions
BENCHMARK_MODELS = [
    # YOLOv8 Series - Core models
    ModelConfig("YOLOv8n", "yolov8n.pt", "tiny", "3.2M", "Fastest - edge deployment"),
    ModelConfig("YOLOv8s", "yolov8s.pt", "small", "11.2M", "Small - balanced"),
    ModelConfig("YOLOv8m", "yolov8m.pt", "medium", "25.9M", "Medium - baseline"),
    ModelConfig("YOLOv8l", "yolov8l.pt", "large", "43.7M", "Large - high accuracy"),
    ModelConfig("YOLOv8x", "yolov8x.pt", "xlarge", "68.2M", "Largest - max accuracy"),
    
    # YOLOv9 Series
    ModelConfig("YOLOv9t", "yolov9t.pt", "tiny", "2.0M", "Ultra efficient"),
    ModelConfig("YOLOv9s", "yolov9s.pt", "small", "7.2M", "Small improved"),
    ModelConfig("YOLOv9m", "yolov9m.pt", "medium", "20.1M", "Medium balanced"),
    ModelConfig("YOLOv9c", "yolov9c.pt", "medium", "25.5M", "Compact optimized"),
    ModelConfig("YOLOv9e", "yolov9e.pt", "large", "58.1M", "Enhanced accuracy"),
    
    # YOLOv10 Series
    ModelConfig("YOLOv10n", "yolov10n.pt", "tiny", "2.3M", "Most efficient"),
    ModelConfig("YOLOv10s", "yolov10s.pt", "small", "7.2M", "Fast inference"),
    ModelConfig("YOLOv10m", "yolov10m.pt", "medium", "15.4M", "Production ready"),
    ModelConfig("YOLOv10l", "yolov10l.pt", "large", "24.4M", "High accuracy"),
    ModelConfig("YOLOv10x", "yolov10x.pt", "xlarge", "29.5M", "Maximum v10"),
    
    # YOLOv11 Series - Latest
    ModelConfig("YOLO11n", "yolo11n.pt", "tiny", "2.6M", "Latest nano"),
    ModelConfig("YOLO11s", "yolo11s.pt", "small", "9.4M", "Latest small"),
    ModelConfig("YOLO11m", "yolo11m.pt", "medium", "20.1M", "Latest medium"),
    ModelConfig("YOLO11l", "yolo11l.pt", "large", "25.3M", "Latest large"),
    ModelConfig("YOLO11x", "yolo11x.pt", "xlarge", "56.9M", "Latest maximum"),
]

class CustomDatasetBenchmark:
    """Benchmark system specifically for your custom dataset"""
    
    def __init__(self, dataset_path, max_images=None):
        self.dataset_path = Path(dataset_path)
        self.max_images = max_images  # Limit for faster testing
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.results = []
        
        # Load dataset configuration
        self.dataset_config = self.load_dataset_config()
        self.class_names = self.dataset_config.get('names', ['person'])
        self.num_classes = self.dataset_config.get('nc', 1)
        
        # Find test images and labels
        self.test_images_path = self.dataset_path / "test" / "images"
        self.test_labels_path = self.dataset_path / "test" / "labels"
        
        self.image_files = self.get_test_images()
        self.label_files = self.get_test_labels()
        
        print(f"Custom Dataset Benchmark initialized")
        print(f"Dataset: {self.dataset_path}")
        print(f"Device: {self.device.upper()}")
        print(f"Classes: {self.class_names}")
        print(f"Test images: {len(self.image_files)}")
        print(f"Test labels: {len(self.label_files)}")
        if max_images is None:
            print(f"Testing: ALL {len(self.image_files)} images for complete accuracy")
        else:
            print(f"Limited to: {max_images} images for speed")
    
    def load_dataset_config(self):
        """Load data.yaml configuration"""
        config_path = self.dataset_path / "data.yaml"
        
        if not config_path.exists():
            print(f"[WARNING] data.yaml not found at {config_path}")
            return {'names': ['person'], 'nc': 1}
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            print(f"[SUCCESS] Loaded dataset config:")
            print(f"  Classes: {config.get('names', 'Not defined')}")
            print(f"  Count: {config.get('nc', 'Not defined')}")
            
            return config
            
        except Exception as e:
            print(f"[ERROR] Failed to load data.yaml: {e}")
            return {'names': ['person'], 'nc': 1}
    
    def get_test_images(self):
        """Get list of test images"""
        if not self.test_images_path.exists():
            print(f"[ERROR] Test images path not found: {self.test_images_path}")
            return []
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(self.test_images_path.glob(f'*{ext}')))
            image_files.extend(list(self.test_images_path.glob(f'*{ext.upper()}')))
        
        # Sort for consistent ordering
        image_files = sorted(image_files)
        
        # Limit if specified
        if self.max_images:
            image_files = image_files[:self.max_images]
        
        return image_files
    
    def get_test_labels(self):
        """Get list of test labels"""
        if not self.test_labels_path.exists():
            print(f"[ERROR] Test labels path not found: {self.test_labels_path}")
            return []
        
        label_files = list(self.test_labels_path.glob('*.txt'))
        return sorted(label_files)
    
    def load_ground_truth(self, image_file):
        """Load ground truth annotations for an image"""
        # Find corresponding label file
        image_stem = image_file.stem
        label_file = self.test_labels_path / f"{image_stem}.txt"
        
        if not label_file.exists():
            return []
        
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            ground_truth = []
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        
                        ground_truth.append({
                            'class_id': class_id,
                            'bbox': [x_center, y_center, width, height],
                            'format': 'yolo_normalized'
                        })
            
            return ground_truth
            
        except Exception as e:
            print(f"[ERROR] Failed to load ground truth for {image_file.name}: {e}")
            return []
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes in YOLO format (x_center, y_center, width, height)"""
        # Convert to corner coordinates
        def yolo_to_corners(x_center, y_center, width, height):
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            return x1, y1, x2, y2
        
        x1_1, y1_1, x2_1, y2_1 = yolo_to_corners(*box1)
        x1_2, y1_2, x2_2, y2_2 = yolo_to_corners(*box2)
        
        # Calculate intersection
        x1_int = max(x1_1, x1_2)
        y1_int = max(y1_1, y1_2)
        x2_int = min(x2_1, x2_2)
        y2_int = min(y2_1, y2_2)
        
        if x2_int <= x1_int or y2_int <= y1_int:
            return 0.0
        
        intersection = (x2_int - x1_int) * (y2_int - y1_int)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union <= 0:
            return 0.0
        
        return intersection / union
    
    def calculate_map(self, all_predictions, all_ground_truths, iou_threshold=0.5):
        """Calculate mAP@0.5 for the dataset"""
        if not all_predictions or not all_ground_truths:
            return 0.0, {}
        
        # Collect all predictions with confidence scores
        all_preds = []
        for preds in all_predictions:
            for pred in preds:
                all_preds.append({
                    'class_id': pred['class_id'],
                    'confidence': pred['confidence'],
                    'bbox': pred['bbox'],
                    'image_id': pred.get('image_id', 0)
                })
        
        # Sort by confidence (descending)
        all_preds.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Calculate precision and recall
        true_positives = 0
        false_positives = 0
        total_ground_truth = sum(len(gt) for gt in all_ground_truths)
        
        matched_ground_truth = set()
        
        for pred in all_preds:
            pred_class = pred['class_id']
            pred_bbox = pred['bbox']
            pred_image = pred['image_id']
            
            # Find matching ground truth
            best_iou = 0
            best_gt_idx = -1
            
            if pred_image < len(all_ground_truths):
                for gt_idx, gt in enumerate(all_ground_truths[pred_image]):
                    if gt['class_id'] == pred_class:
                        iou = self.calculate_iou(pred_bbox, gt['bbox'])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
            
            # Check if it's a true positive
            gt_key = (pred_image, best_gt_idx)
            if best_iou >= iou_threshold and gt_key not in matched_ground_truth:
                true_positives += 1
                matched_ground_truth.add(gt_key)
            else:
                false_positives += 1
        
        # Calculate final metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / total_ground_truth if total_ground_truth > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # mAP approximation (simplified for single class)
        map_score = precision * recall  # Simplified mAP calculation
        
        details = {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'total_ground_truth': total_ground_truth,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'predictions_count': len(all_preds)
        }
        
        return map_score, details
    
    def get_model_info(self, model):
        """Get model parameter count and size"""
        try:
            param_count = sum(p.numel() for p in model.model.parameters())
            param_str = f"{param_count/1e6:.1f}M"
            return param_count, param_str
        except:
            return 0, "Unknown"
    
    def get_memory_usage(self):
        """Get current GPU and RAM memory usage"""
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024**3)
        
        ram_memory = psutil.virtual_memory().used / (1024**3)
        return gpu_memory, ram_memory
    
    def benchmark_model(self, model_config):
        """Benchmark a single model on your custom dataset"""
        print(f"\n[TESTING] {model_config.name} ({model_config.category.upper()})")
        print(f"Expected params: {model_config.expected_params}")
        
        result = {
            'name': model_config.name,
            'category': model_config.category,
            'expected_params': model_config.expected_params,
            'notes': model_config.notes,
            'success': False,
            'error': None,
            'metrics': {}
        }
        
        try:
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Record initial memory
            initial_gpu, initial_ram = self.get_memory_usage()
            
            # Load model
            print(f"  Loading {model_config.model_file}...")
            start_load = time.time()
            model = YOLO(model_config.model_file)
            if self.device == 'cuda':
                model.to(self.device)
            load_time = time.time() - start_load
            
            # Get model info
            param_count, param_str = self.get_model_info(model)
            
            # Record memory after model loading
            post_load_gpu, post_load_ram = self.get_memory_usage()
            
            # Run inference on test images
            all_predictions = []
            all_ground_truths = []
            inference_times = []
            total_detections = 0
            images_processed = 0
            
            print(f"  Testing on {len(self.image_files)} images...")
            if len(self.image_files) > 150:
                print(f"  [INFO] Full dataset test - this will take longer but give accurate results")
            
            total_start = time.time()
            
            for img_idx, image_file in enumerate(self.image_files):
                if self.max_images and img_idx >= self.max_images:
                    break
                
                # Load image
                try:
                    image = cv2.imread(str(image_file))
                    if image is None:
                        continue
                    
                    # Load ground truth
                    ground_truth = self.load_ground_truth(image_file)
                    all_ground_truths.append(ground_truth)
                    
                    # Run inference
                    inf_start = time.time()
                    results = model(
                        image,
                        conf=0.25,
                        device=self.device,
                        verbose=False
                    )
                    inf_time = time.time() - inf_start
                    inference_times.append(inf_time)
                    
                    # Extract predictions
                    frame_predictions = []
                    if results[0].boxes is not None:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        confs = results[0].boxes.conf.cpu().numpy()
                        classes = results[0].boxes.cls.cpu().numpy()
                        
                        # Convert to YOLO format for mAP calculation
                        img_height, img_width = image.shape[:2]
                        
                        for box, conf, cls in zip(boxes, confs, classes):
                            x1, y1, x2, y2 = box
                            
                            # Convert to YOLO normalized format
                            x_center = (x1 + x2) / 2 / img_width
                            y_center = (y1 + y2) / 2 / img_height
                            width = (x2 - x1) / img_width
                            height = (y2 - y1) / img_height
                            
                            frame_predictions.append({
                                'class_id': int(cls),
                                'confidence': float(conf),
                                'bbox': [x_center, y_center, width, height],
                                'image_id': img_idx
                            })
                    
                    all_predictions.append(frame_predictions)
                    total_detections += len(frame_predictions)
                    images_processed += 1
                    
                    # Progress update
                    progress_interval = 25 if len(self.image_files) <= 100 else 50
                    if (img_idx + 1) % progress_interval == 0:
                        progress = (img_idx + 1) / len(self.image_files) * 100
                        print(f"    Progress: {img_idx + 1}/{len(self.image_files)} images ({progress:.1f}%)")
                    
                except Exception as e:
                    print(f"    [ERROR] Failed to process {image_file.name}: {e}")
                    continue
            
            total_time = time.time() - total_start
            
            # Calculate metrics
            if inference_times and images_processed > 0:
                # Speed metrics
                avg_inference = np.mean(inference_times)
                avg_fps = 1.0 / avg_inference if avg_inference > 0 else 0
                
                # mAP calculation
                map_score, map_details = self.calculate_map(all_predictions, all_ground_truths)
                
                # Detection metrics
                avg_detections = total_detections / images_processed
                
                # Memory metrics
                peak_gpu, peak_ram = self.get_memory_usage()
                
                # Store results
                result['metrics'] = {
                    # KEY METRICS
                    'parameters': param_count,
                    'fps': avg_fps,
                    'map_50': map_score,
                    'precision': map_details.get('precision', 0),
                    'recall': map_details.get('recall', 0),
                    'f1_score': map_details.get('f1_score', 0),
                    'memory_gb': peak_gpu,
                    
                    # Additional metrics
                    'param_str': param_str,
                    'load_time': load_time,
                    'avg_inference_time': avg_inference,
                    'images_processed': images_processed,
                    'total_detections': total_detections,
                    'avg_detections_per_image': avg_detections,
                    'ground_truth_objects': map_details.get('total_ground_truth', 0),
                    'true_positives': map_details.get('true_positives', 0),
                    'false_positives': map_details.get('false_positives', 0),
                    'model_memory_usage_gb': post_load_gpu - initial_gpu,
                    'peak_memory_gb': peak_gpu,
                    
                    # Dataset specific
                    'dataset_images': len(self.image_files),
                    'dataset_labels': len(self.label_files),
                    'class_names': self.class_names,
                    'num_classes': self.num_classes
                }
                
                result['success'] = True
                
                print(f"  [SUCCESS] {model_config.name}")
                print(f"    FPS: {avg_fps:.1f} | mAP@0.5: {map_score:.3f} | Precision: {map_details.get('precision', 0):.3f}")
                print(f"    Recall: {map_details.get('recall', 0):.3f} | F1: {map_details.get('f1_score', 0):.3f}")
                print(f"    Detections: {avg_detections:.1f}/image | Memory: {peak_gpu:.1f}GB")
                
            else:
                result['error'] = "No valid inference data collected"
                
        except Exception as e:
            result['error'] = str(e)
            print(f"  [ERROR] {model_config.name}: {e}")
            
            # Common error solutions
            error_str = str(e).lower()
            if "out of memory" in error_str:
                print(f"    [SOLUTION] Reduce max_images or use smaller model")
            elif "model not found" in error_str:
                print(f"    [SOLUTION] Model will be downloaded automatically")
        
        # Clean up
        try:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except:
            pass
        
        return result
    
    def run_benchmark(self, models_to_test=None):
        """Run benchmark on selected models"""
        if models_to_test is None:
            models_to_test = BENCHMARK_MODELS
        
        print_header()
        print(f"Testing {len(models_to_test)} models on your custom dataset...")
        print(f"Dataset: {self.dataset_config.get('names', ['Unknown'])}")
        print(f"Test images: {len(self.image_files)}")
        print(f"Ground truth labels: {len(self.label_files)}")
        
        start_time = time.time()
        
        for i, model_config in enumerate(models_to_test, 1):
            print(f"\n--- Model {i}/{len(models_to_test)} ---")
            result = self.benchmark_model(model_config)
            self.results.append(result)
        
        total_time = time.time() - start_time
        
        print(f"\n[COMPLETED] Benchmark finished in {total_time:.1f}s")
        
        # Generate report
        self.generate_report()
        self.save_results()
        
        return self.results
    
    def generate_report(self):
        """Generate comprehensive benchmark report"""
        print_statistics_header()
        
        successful = [r for r in self.results if r['success']]
        failed = [r for r in self.results if not r['success']]
        
        print(f"Models tested: {len(self.results)} | Successful: {len(successful)} | Failed: {len(failed)}")
        print(f"Dataset: {self.class_names} | Test images: {len(self.image_files)} | Labels: {len(self.label_files)}")
        
        if failed:
            print(f"\n[FAILED MODELS]")
            for result in failed:
                print(f"  {result['name']}: {result['error']}")
        
        if successful:
            print(f"\n{'='*100}")
            print(f"CUSTOM DATASET BENCHMARK RESULTS")
            print(f"{'='*100}")
            
            # Main results table
            print(f"\n{'Model':<15} {'Cat':<8} {'Params':<8} {'FPS':<7} {'mAP@0.5':<8} {'Precision':<10} {'Recall':<8} {'F1':<6} {'Memory':<8}")
            print(f"{'-'*90}")
            
            # Sort by mAP (accuracy)
            for result in sorted(successful, key=lambda x: x['metrics']['map_50'], reverse=True):
                m = result['metrics']
                
                print(f"{result['name']:<15} {result['category'][:6]:<8} {m['param_str']:<8} "
                      f"{m['fps']:<7.1f} {m['map_50']:<8.3f} {m['precision']:<10.3f} "
                      f"{m['recall']:<8.3f} {m['f1_score']:<6.3f} {m['memory_gb']:<8.1f}")
            
            # Top performers
            print(f"\n{'='*60}")
            print(f"TOP PERFORMERS ON YOUR DATASET")
            print(f"{'='*60}")
            
            best_accuracy = max(successful, key=lambda x: x['metrics']['map_50'])
            fastest = max(successful, key=lambda x: x['metrics']['fps'])
            most_efficient = min(successful, key=lambda x: x['metrics']['memory_gb'])
            best_f1 = max(successful, key=lambda x: x['metrics']['f1_score'])
            
            print(f"[BEST ACCURACY]  {best_accuracy['name']} (mAP@0.5: {best_accuracy['metrics']['map_50']:.3f})")
            print(f"[FASTEST]        {fastest['name']} ({fastest['metrics']['fps']:.1f} FPS)")
            print(f"[MOST EFFICIENT] {most_efficient['name']} ({most_efficient['metrics']['memory_gb']:.1f}GB)")
            print(f"[BEST F1 SCORE]  {best_f1['name']} (F1: {best_f1['metrics']['f1_score']:.3f})")
            
            # Performance analysis
            print(f"\n{'='*60}")
            print(f"PERFORMANCE ANALYSIS")
            print(f"{'='*60}")
            
            avg_map = np.mean([r['metrics']['map_50'] for r in successful])
            avg_fps = np.mean([r['metrics']['fps'] for r in successful])
            avg_memory = np.mean([r['metrics']['memory_gb'] for r in successful])
            
            print(f"Average mAP@0.5: {avg_map:.3f}")
            print(f"Average FPS: {avg_fps:.1f}")
            print(f"Average Memory: {avg_memory:.1f}GB")
            
            # Real-time capable models (>30 FPS)
            realtime = [r for r in successful if r['metrics']['fps'] >= 30]
            print(f"\nReal-time capable (≥30 FPS): {len(realtime)} models")
            if realtime:
                for model in sorted(realtime, key=lambda x: x['metrics']['map_50'], reverse=True)[:3]:
                    print(f"  {model['name']}: {model['metrics']['fps']:.1f} FPS, mAP {model['metrics']['map_50']:.3f}")
            
            # High accuracy models (top 25%)
            accuracy_threshold = np.percentile([r['metrics']['map_50'] for r in successful], 75)
            high_accuracy = [r for r in successful if r['metrics']['map_50'] >= accuracy_threshold]
            print(f"\nHigh accuracy (top 25%): {len(high_accuracy)} models")
            if high_accuracy:
                for model in sorted(high_accuracy, key=lambda x: x['metrics']['fps'], reverse=True)[:3]:
                    print(f"  {model['name']}: mAP {model['metrics']['map_50']:.3f}, {model['metrics']['fps']:.1f} FPS")
            
            # Deployment recommendations
            print(f"\n{'='*60}")
            print(f"DEPLOYMENT RECOMMENDATIONS FOR YOUR DATASET")
            print(f"{'='*60}")
            
            # Edge deployment (fast + efficient)
            edge_models = [r for r in successful if r['metrics']['fps'] >= 25 and r['metrics']['memory_gb'] <= 4]
            if edge_models:
                best_edge = max(edge_models, key=lambda x: x['metrics']['map_50'])
                print(f"[EDGE DEPLOYMENT]")
                print(f"  Recommended: {best_edge['name']}")
                print(f"  Performance: {best_edge['metrics']['fps']:.1f} FPS, {best_edge['metrics']['memory_gb']:.1f}GB")
                print(f"  Accuracy: mAP@0.5 {best_edge['metrics']['map_50']:.3f}")
            
            # Production deployment (balanced)
            balanced = [r for r in successful if r['metrics']['fps'] >= 20 and r['metrics']['map_50'] >= avg_map]
            if balanced:
                best_balanced = max(balanced, key=lambda x: x['metrics']['f1_score'])
                print(f"\n[PRODUCTION DEPLOYMENT]")
                print(f"  Recommended: {best_balanced['name']}")
                print(f"  Performance: {best_balanced['metrics']['fps']:.1f} FPS")
                print(f"  Accuracy: mAP@0.5 {best_balanced['metrics']['map_50']:.3f}, F1 {best_balanced['metrics']['f1_score']:.3f}")
            
            # Maximum accuracy (regardless of speed)
            if successful:
                max_accuracy = max(successful, key=lambda x: x['metrics']['map_50'])
                print(f"\n[MAXIMUM ACCURACY]")
                print(f"  Recommended: {max_accuracy['name']}")
                print(f"  Accuracy: mAP@0.5 {max_accuracy['metrics']['map_50']:.3f}")
                print(f"  Performance: {max_accuracy['metrics']['fps']:.1f} FPS, {max_accuracy['metrics']['memory_gb']:.1f}GB")
    
    def save_results(self):
        """Save detailed results to JSON"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"custom_dataset_benchmark_{timestamp}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': str(self.dataset_path),
            'dataset_config': self.dataset_config,
            'test_images_count': len(self.image_files),
            'test_labels_count': len(self.label_files),
            'device': self.device,
            'models_tested': len(self.results),
            'successful_tests': len([r for r in self.results if r['success']]),
            'results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n[SAVED] Detailed results: {filename}")

# Main execution
if __name__ == "__main__":
    # Configuration
    DATASET_PATH = r"D:\User\WS\DATPAPA\DAT\DATA.v1i.yoloProject"
    MAX_IMAGES = None  # Test ALL 236 images for complete accuracy
    
    # Select models to test
    models_to_test = [
        # ALL MODELS - Complete benchmark across all YOLO series
        *BENCHMARK_MODELS  # All 20 models: YOLOv8, v9, v10, v11 series
    ]
    
    print(f"CUSTOM DATASET BENCHMARK")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Max images: {'All 236 images' if MAX_IMAGES is None else MAX_IMAGES}")
    print(f"Models to test: {len(models_to_test)}")
    print(f"GPU acceleration: {'ENABLED' if torch.cuda.is_available() else 'DISABLED'}")
    
    try:
        benchmark = CustomDatasetBenchmark(DATASET_PATH, MAX_IMAGES)
        
        if MAX_IMAGES is None and len(benchmark.image_files) > 150:
            print(f"\n[INFO] Full dataset benchmark - Estimated time: 45-90 minutes")
            print(f"[INFO] Testing all {len(benchmark.image_files)} images for maximum accuracy")
        
        results = benchmark.run_benchmark(models_to_test)
        
        print(f"\n[COMPLETED] Custom dataset benchmark complete!")
        print(f"Check the results above and saved JSON file for detailed metrics.")
        
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Benchmark stopped by user")
    except Exception as e:
        print(f"[ERROR] Benchmark failed: {e}")