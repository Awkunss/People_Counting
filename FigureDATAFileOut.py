"""
Dataset Explorer for DATA.v1i.yoloProject
This script analyzes your specific YOLO dataset to understand the structure for benchmarking
"""

import os
import sys
from pathlib import Path
import json
import yaml
from datetime import datetime
import cv2
import numpy as np
from collections import defaultdict, Counter
import glob

def print_header():
    """Print professional header"""
    print("=" * 80)
    print("DATASET EXPLORER - DATA.v1i.yoloProject")
    print("Understanding Your Custom Dataset for Benchmark Modification")
    print("=" * 80)

class DatasetExplorer:
    """Focused exploration of your specific dataset"""
    
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.analysis = {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': str(self.dataset_path.absolute()),
            'config': {},
            'test_images': {},
            'test_labels': {},
            'other_splits': {},
            'class_info': {},
            'benchmark_config': {}
        }
        
        print(f"Exploring dataset: {self.dataset_path.absolute()}")
        print()
    
    def get_file_size_str(self, size_bytes):
        """Convert bytes to human readable format"""
        if size_bytes == 0:
            return "0 B"
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024.0 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        return f"{size_bytes:.1f} {size_names[i]}"
    
    def analyze_data_yaml(self):
        """Analyze the data.yaml configuration file"""
        print("[STEP 1] ANALYZING data.yaml")
        print("=" * 60)
        
        data_yaml_path = self.dataset_path / "data.yaml"
        
        if not data_yaml_path.exists():
            print("[ERROR] data.yaml not found!")
            return {}
        
        try:
            with open(data_yaml_path, 'r') as f:
                yaml_content = yaml.safe_load(f)
            
            print("[SUCCESS] data.yaml loaded successfully")
            print("\nContents:")
            
            config = {}
            for key, value in yaml_content.items():
                print(f"  {key}: {value}")
                config[key] = value
            
            # Extract and analyze class information
            if 'names' in yaml_content:
                classes = yaml_content['names']
                
                if isinstance(classes, dict):
                    config['class_mapping'] = classes
                    config['num_classes'] = len(classes)
                    print(f"\n[CLASSES] Found {len(classes)} classes:")
                    for class_id, class_name in classes.items():
                        print(f"  ID {class_id}: {class_name}")
                        
                elif isinstance(classes, list):
                    config['class_mapping'] = {i: name for i, name in enumerate(classes)}
                    config['num_classes'] = len(classes)
                    print(f"\n[CLASSES] Found {len(classes)} classes:")
                    for i, class_name in enumerate(classes):
                        print(f"  ID {i}: {class_name}")
            
            # Check dataset paths
            for split_key in ['train', 'val', 'test']:
                if split_key in yaml_content:
                    split_path = yaml_content[split_key]
                    full_path = self.dataset_path / split_path if not Path(split_path).is_absolute() else Path(split_path)
                    exists = full_path.exists()
                    status = "[EXISTS]" if exists else "[MISSING]"
                    print(f"  {split_key}: {split_path} {status}")
                    config[f'{split_key}_path'] = split_path
                    config[f'{split_key}_exists'] = exists
            
            self.analysis['config'] = config
            print()
            return config
            
        except Exception as e:
            print(f"[ERROR] Failed to read data.yaml: {e}")
            return {}
    
    def analyze_test_images(self):
        """Analyze images in test/images folder"""
        print("[STEP 2] ANALYZING test/images/")
        print("=" * 60)
        
        test_images_path = self.dataset_path / "test" / "images"
        
        if not test_images_path.exists():
            print("[ERROR] test/images/ folder not found!")
            return {}
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(test_images_path.glob(f'*{ext}')))
            image_files.extend(list(test_images_path.glob(f'*{ext.upper()}')))
        
        print(f"[FOUND] {len(image_files)} test images")
        
        if not image_files:
            print("[ERROR] No image files found in test/images/")
            return {}
        
        # Analyze sample images
        sample_size = min(10, len(image_files))
        sample_images = image_files[:sample_size]
        
        resolutions = []
        file_sizes = []
        valid_images = []
        
        print(f"\n[ANALYZING] First {sample_size} images:")
        
        for i, img_file in enumerate(sample_images):
            try:
                img = cv2.imread(str(img_file))
                if img is not None:
                    height, width, channels = img.shape
                    file_size = img_file.stat().st_size
                    
                    resolutions.append((width, height))
                    file_sizes.append(file_size)
                    valid_images.append({
                        'filename': img_file.name,
                        'width': width,
                        'height': height,
                        'size': self.get_file_size_str(file_size)
                    })
                    
                    print(f"  {i+1}. {img_file.name}: {width}x{height}, {self.get_file_size_str(file_size)}")
                else:
                    print(f"  {i+1}. {img_file.name}: [ERROR] Cannot read image")
                    
            except Exception as e:
                print(f"  {i+1}. {img_file.name}: [ERROR] {e}")
        
        # Image statistics
        analysis = {
            'total_images': len(image_files),
            'analyzed_images': len(valid_images),
            'all_filenames': [f.name for f in image_files],
            'sample_analysis': valid_images
        }
        
        if resolutions:
            unique_resolutions = list(set(resolutions))
            most_common_res = Counter(resolutions).most_common(1)[0]
            
            analysis.update({
                'unique_resolutions': len(unique_resolutions),
                'resolutions': unique_resolutions,
                'most_common_resolution': most_common_res,
                'width_range': f"{min(r[0] for r in resolutions)}-{max(r[0] for r in resolutions)}",
                'height_range': f"{min(r[1] for r in resolutions)}-{max(r[1] for r in resolutions)}",
                'total_size': self.get_file_size_str(sum(file_sizes)) if file_sizes else "0 B"
            })
            
            print(f"\n[SUMMARY]")
            print(f"  Total images: {len(image_files)}")
            print(f"  Unique resolutions: {len(unique_resolutions)}")
            print(f"  Most common: {most_common_res[0]} ({most_common_res[1]} images)")
            print(f"  Size range: {analysis['width_range']} x {analysis['height_range']}")
            print(f"  Total size: {analysis['total_size']}")
        
        self.analysis['test_images'] = analysis
        print()
        return analysis
    
    def analyze_test_labels(self):
        """Analyze labels in test/labels folder"""
        print("[STEP 3] ANALYZING test/labels/")
        print("=" * 60)
        
        test_labels_path = self.dataset_path / "test" / "labels"
        
        if not test_labels_path.exists():
            print("[ERROR] test/labels/ folder not found!")
            return {}
        
        # Find all label files
        label_files = list(test_labels_path.glob('*.txt'))
        
        print(f"[FOUND] {len(label_files)} label files")
        
        if not label_files:
            print("[ERROR] No label files found in test/labels/")
            return {}
        
        # Analyze sample label files
        sample_size = min(10, len(label_files))
        sample_labels = label_files[:sample_size]
        
        class_counts = defaultdict(int)
        objects_per_image = []
        annotation_quality = []
        
        print(f"\n[ANALYZING] First {sample_size} label files:")
        
        for i, label_file in enumerate(sample_labels):
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                objects_in_file = 0
                classes_in_file = set()
                valid_annotations = 0
                
                for line in lines:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            try:
                                class_id = int(parts[0])
                                x, y, w, h = map(float, parts[1:5])
                                
                                # Validate YOLO format (normalized coordinates)
                                if 0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1:
                                    valid_annotations += 1
                                    objects_in_file += 1
                                    classes_in_file.add(class_id)
                                    class_counts[class_id] += 1
                                
                            except ValueError:
                                pass
                
                objects_per_image.append(objects_in_file)
                annotation_quality.append({
                    'filename': label_file.name,
                    'total_lines': len(lines),
                    'valid_objects': objects_in_file,
                    'classes': list(classes_in_file)
                })
                
                print(f"  {i+1}. {label_file.name}: {objects_in_file} objects, classes {list(classes_in_file)}")
                
            except Exception as e:
                print(f"  {i+1}. {label_file.name}: [ERROR] {e}")
        
        # Label statistics
        analysis = {
            'total_labels': len(label_files),
            'analyzed_labels': len(annotation_quality),
            'all_filenames': [f.name for f in label_files],
            'sample_analysis': annotation_quality
        }
        
        if objects_per_image:
            analysis.update({
                'class_distribution': dict(class_counts),
                'unique_classes': list(class_counts.keys()),
                'total_objects': sum(objects_per_image),
                'avg_objects_per_image': np.mean(objects_per_image),
                'min_objects_per_image': min(objects_per_image),
                'max_objects_per_image': max(objects_per_image),
                'objects_per_image_distribution': objects_per_image
            })
            
            print(f"\n[SUMMARY]")
            print(f"  Total labels: {len(label_files)}")
            print(f"  Total objects: {analysis['total_objects']}")
            print(f"  Avg objects/image: {analysis['avg_objects_per_image']:.1f}")
            print(f"  Objects range: {analysis['min_objects_per_image']}-{analysis['max_objects_per_image']}")
            print(f"  Classes found: {sorted(analysis['unique_classes'])}")
            
            print(f"\n[CLASS DISTRIBUTION]")
            for class_id, count in sorted(class_counts.items()):
                class_name = "Unknown"
                if 'class_mapping' in self.analysis['config']:
                    class_name = self.analysis['config']['class_mapping'].get(class_id, f"Class_{class_id}")
                print(f"  Class {class_id} ({class_name}): {count} objects")
        
        self.analysis['test_labels'] = analysis
        print()
        return analysis
    
    def check_other_splits(self):
        """Check for train/val folders or other dataset splits"""
        print("[STEP 4] CHECKING OTHER DATASET SPLITS")
        print("=" * 60)
        
        other_splits = {}
        
        # Check for train and val folders
        for split_name in ['train', 'val', 'valid']:
            split_path = self.dataset_path / split_name
            
            if split_path.exists() and split_path.is_dir():
                print(f"[FOUND] {split_name}/ folder")
                
                images_path = split_path / 'images'
                labels_path = split_path / 'labels'
                
                split_info = {'exists': True}
                
                if images_path.exists():
                    image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
                    split_info['images_count'] = len(image_files)
                    print(f"  Images: {len(image_files)} files")
                
                if labels_path.exists():
                    label_files = list(labels_path.glob('*.txt'))
                    split_info['labels_count'] = len(label_files)
                    print(f"  Labels: {len(label_files)} files")
                
                other_splits[split_name] = split_info
            else:
                print(f"[NOT FOUND] {split_name}/ folder")
        
        # Check for alternative structures
        images_root = self.dataset_path / 'images'
        labels_root = self.dataset_path / 'labels'
        
        if images_root.exists() and labels_root.exists():
            print(f"\n[FOUND] Alternative structure: images/ and labels/ in root")
            
            for split_name in ['train', 'val', 'test']:
                split_images = images_root / split_name
                split_labels = labels_root / split_name
                
                if split_images.exists() or split_labels.exists():
                    alt_info = {}
                    if split_images.exists():
                        img_count = len(list(split_images.glob('*.jpg')) + list(split_images.glob('*.png')))
                        alt_info['images_count'] = img_count
                        print(f"  images/{split_name}: {img_count} files")
                    
                    if split_labels.exists():
                        lbl_count = len(list(split_labels.glob('*.txt')))
                        alt_info['labels_count'] = lbl_count
                        print(f"  labels/{split_name}: {lbl_count} files")
                    
                    other_splits[f'{split_name}_alt'] = alt_info
        
        self.analysis['other_splits'] = other_splits
        print()
        return other_splits
    
    def generate_benchmark_config(self):
        """Generate configuration for benchmark modification"""
        print("[STEP 5] GENERATING BENCHMARK CONFIGURATION")
        print("=" * 60)
        
        config = self.analysis['config']
        test_images = self.analysis['test_images']
        test_labels = self.analysis['test_labels']
        
        benchmark_config = {
            'dataset_path': str(self.dataset_path),
            'dataset_name': self.dataset_path.name,
            'ready_for_benchmark': False,
            'issues': [],
            'modifications_needed': []
        }
        
        # Check if dataset is ready for benchmarking
        issues = []
        modifications = []
        
        # Check data.yaml
        if not config:
            issues.append("data.yaml missing or invalid")
        else:
            if 'class_mapping' in config:
                benchmark_config['classes'] = config['class_mapping']
                benchmark_config['num_classes'] = config['num_classes']
                print(f"[OK] Classes defined: {config['num_classes']}")
            else:
                issues.append("Class definitions missing in data.yaml")
        
        # Check test images
        if not test_images or test_images.get('total_images', 0) == 0:
            issues.append("No test images found")
        else:
            benchmark_config['test_images_count'] = test_images['total_images']
            benchmark_config['test_image_path'] = str(self.dataset_path / "test" / "images")
            print(f"[OK] Test images: {test_images['total_images']}")
        
        # Check test labels
        if not test_labels or test_labels.get('total_labels', 0) == 0:
            issues.append("No test labels found")
        else:
            benchmark_config['test_labels_count'] = test_labels['total_labels']
            benchmark_config['test_labels_path'] = str(self.dataset_path / "test" / "labels")
            print(f"[OK] Test labels: {test_labels['total_labels']}")
            
            # Check image/label balance
            img_count = test_images.get('total_images', 0)
            lbl_count = test_labels.get('total_labels', 0)
            
            if img_count > 0 and lbl_count > 0:
                balance_ratio = lbl_count / img_count
                if 0.8 <= balance_ratio <= 1.2:
                    print(f"[OK] Good image/label balance: {balance_ratio:.2f}")
                else:
                    issues.append(f"Image/label imbalance: {balance_ratio:.2f}")
        
        # Determine modifications needed
        if 'classes' in benchmark_config:
            modifications.append("Update benchmark to use custom classes instead of generic 'person'")
            modifications.append("Load data.yaml for class definitions")
            modifications.append("Calculate mAP using ground truth labels")
        
        if test_images and test_labels:
            modifications.append("Replace video-based testing with image dataset evaluation")
            modifications.append("Add proper YOLO evaluation metrics")
            modifications.append("Compare custom trained models with pretrained models")
        
        # Overall readiness
        if len(issues) == 0:
            benchmark_config['ready_for_benchmark'] = True
            print(f"\n[SUCCESS] Dataset ready for benchmark modification!")
        else:
            print(f"\n[ISSUES] Dataset has {len(issues)} issues:")
            for issue in issues:
                print(f"  - {issue}")
        
        benchmark_config['issues'] = issues
        benchmark_config['modifications_needed'] = modifications
        
        print(f"\n[MODIFICATIONS NEEDED]")
        for mod in modifications:
            print(f"  - {mod}")
        
        self.analysis['benchmark_config'] = benchmark_config
        print()
        return benchmark_config
    
    def save_analysis(self):
        """Save analysis to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"dataset_exploration_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.analysis, f, indent=2, default=str)
        
        print(f"[SAVED] Analysis saved to: {filename}")
        return filename
    
    def run_exploration(self):
        """Run complete dataset exploration"""
        print_header()
        
        try:
            # Run all exploration steps
            self.analyze_data_yaml()
            self.analyze_test_images()
            self.analyze_test_labels()
            self.check_other_splits()
            self.generate_benchmark_config()
            
            # Save results
            report_file = self.save_analysis()
            
            print("=" * 80)
            print("[EXPLORATION COMPLETE]")
            print("=" * 80)
            print(f"Dataset: {self.dataset_path.name}")
            print(f"Ready for benchmark: {self.analysis['benchmark_config']['ready_for_benchmark']}")
            print(f"Report saved: {report_file}")
            print("\nReady to modify benchmark code with this information!")
            
            return self.analysis
            
        except Exception as e:
            print(f"\n[ERROR] Exploration failed: {e}")
            return None

# Main execution
if __name__ == "__main__":
    # Your dataset path
    DATASET_PATH = r"D:\User\WS\DATPAPA\DAT\DATA.v1i.yoloProject"
    
    print(f"Dataset Exploration")
    print(f"Target: {DATASET_PATH}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        explorer = DatasetExplorer(DATASET_PATH)
        results = explorer.run_exploration()
        
        if results and results['benchmark_config']['ready_for_benchmark']:
            print("NEXT STEP: Ready to modify benchmark code!")
        else:
            print("NEXT STEP: Fix dataset issues then modify benchmark code.")
        
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Exploration stopped by user")
    except Exception as e:
        print(f"[ERROR] Exploration failed: {e}")
        print("Please check the dataset path and permissions")