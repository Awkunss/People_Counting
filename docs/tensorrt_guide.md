# TensorRT Quantization Guide

## Tổng quan

Hệ thống People Counting hỗ trợ TensorRT quantization để tối ưu performance trên GPU NVIDIA. TensorRT có thể tăng tốc inference lên 2-5x so với PyTorch models.

## Các loại Quantization

### 1. FP32 (Float32)
- **Độ chính xác**: Highest
- **Tốc độ**: Slowest
- **Kích thước**: Largest
- **Sử dụng**: Baseline, debugging

### 2. FP16 (Float16)
- **Độ chính xác**: Very high (~99% FP32)
- **Tốc độ**: 1.5-2x faster than FP32
- **Kích thước**: 50% smaller
- **Sử dụng**: **Recommended** - best balance

### 3. INT8 (Integer8)
- **Độ chính xác**: Good (with calibration)
- **Tốc độ**: 2-4x faster than FP32
- **Kích thước**: 75% smaller
- **Sử dụng**: Maximum performance

## Installation

### Requirements
- NVIDIA GPU (Compute Capability 6.0+)
- CUDA 11.8+ hoặc 12.x
- Python 3.8+

### Setup TensorRT
```bash
# Chạy setup script
python scripts/setup_tensorrt.py

# Hoặc manual install
pip install nvidia-tensorrt
pip install pycuda
```

### Verify Installation
```bash
python test_tensorrt.py
```

## Quantization Process

### 1. Single Model Quantization

#### FP16 (Recommended)
```bash
python scripts/quantize_models.py --model yolov9s.pt --precision fp16
```

#### INT8 (với calibration)
```bash
python scripts/quantize_models.py --model yolov9s.pt --precision int8 --calibration-video Test.mp4
```

### 2. Batch Quantization
```bash
# Quantize tất cả models
python scripts/quantize_models.py --all-models --precision fp16 --benchmark
```

### 3. Custom Parameters
```bash
python scripts/quantize_models.py \
    --model yolov9s.pt \
    --precision fp16 \
    --workspace 2048 \
    --benchmark \
    --calibration-video Custom.mp4
```

## Usage trong Application

### 1. Automatic Detection
System tự động detect TensorRT engines:
```python
# File: yolov9s_fp16.engine sẽ được ưu tiên hơn yolov9s.pt
model = "yolov9s.pt"  # Sẽ dùng yolov9s_fp16.engine nếu có
```

### 2. Explicit Usage
```python
# Chỉ định engine cụ thể
model = "yolov9s_fp16.engine"
```

### 3. Web Interface
Trong web interface, chọn model với extension `.engine`:
- `yolov9s_fp16.engine`
- `yolov8s_fp16.engine`

## Performance Comparison

### YOLOv9s trên RTX 4080
| Model Type | Inference Time | FPS | Accuracy Loss |
|------------|---------------|-----|---------------|
| PyTorch (.pt) | 15.2ms | 65.8 | Baseline |
| TensorRT FP16 | 8.7ms | 114.9 | <1% |
| TensorRT INT8 | 5.3ms | 188.7 | <3% |

### Memory Usage
| Model Type | Model Size | GPU Memory |
|------------|------------|------------|
| PyTorch | 25.1 MB | 1.2 GB |
| TensorRT FP16 | 12.8 MB | 0.8 GB |
| TensorRT INT8 | 6.9 MB | 0.6 GB |

## Configuration

### settings.py
```python
# TensorRT settings
TENSORRT_MODELS = [
    "yolov9s_fp16.engine",
    "yolov8s_fp16.engine"
]
TENSORRT_PRECISION = "fp16"  # Default precision
TENSORRT_WORKSPACE_SIZE = 1024  # MB
```

## Troubleshooting

### Common Issues

#### 1. Import Error
```
ImportError: No module named 'tensorrt'
```
**Solution**: Install TensorRT
```bash
pip install nvidia-tensorrt
```

#### 2. CUDA Error
```
RuntimeError: CUDA out of memory
```
**Solution**: Giảm workspace size
```bash
python scripts/quantize_models.py --workspace 512
```

#### 3. Engine Build Failed
```
Failed to build TensorRT engine
```
**Solutions**:
- Kiểm tra CUDA version compatibility
- Update NVIDIA driver
- Giảm model complexity

#### 4. Low Accuracy với INT8
**Solutions**:
- Sử dụng representative calibration data
- Increase calibration frames
- Use FP16 instead

### Debug Commands
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check TensorRT
python -c "import tensorrt as trt; print(trt.__version__)"

# Check GPU info
nvidia-smi
```

## Best Practices

### 1. Model Selection
- **YOLOv8n**: Fastest, lowest accuracy
- **YOLOv8s**: Good balance (recommended)
- **YOLOv9s**: Best accuracy/speed ratio
- **YOLOv8m/l**: Highest accuracy, slower

### 2. Precision Selection
- **FP16**: Default choice, good balance
- **INT8**: Chỉ khi cần maximum speed
- **FP32**: Chỉ để debugging

### 3. Calibration Data
- Sử dụng data tương tự production
- Ít nhất 100-500 frames
- Diverse scenarios (lighting, angles, etc.)

### 4. Workspace Size
- RTX 30/40 series: 1024-2048 MB
- GTX 16 series: 512-1024 MB
- Older GPUs: 256-512 MB

## Advanced Usage

### Custom Calibration
```python
from app.core.tensorrt_quantizer import create_calibration_data

# Tạo calibration data từ multiple videos
calibration_data = []
videos = ['Video1.mp4', 'Video2.mp4', 'Video3.mp4']

for video in videos:
    data = create_calibration_data(video, num_frames=100)
    calibration_data.extend(data)

# Sử dụng trong quantization
quantizer.quantize_model(
    model_path='yolov9s.pt',
    engine_path='yolov9s_custom.engine',
    precision='int8',
    calibration_data=calibration_data
)
```

### Batch Processing
```python
# Quantize multiple models với different precisions
models = ['yolov8s.pt', 'yolov9s.pt']
precisions = ['fp16', 'int8']

for model in models:
    for precision in precisions:
        quantize_model(model, precision)
```

## Monitoring Performance

### Real-time Metrics
```python
# Trong application
if model_type == 'tensorrt':
    # Log inference time
    start_time = time.time()
    results = model(frame)
    inference_time = time.time() - start_time
    
    logger.info(f"TensorRT inference: {inference_time*1000:.2f}ms")
```

### Benchmark Script
```bash
# Comprehensive benchmark
python scripts/quantize_models.py --model yolov9s.pt --benchmark

# Results sẽ show:
# - Average inference time
# - FPS
# - Memory usage
# - Accuracy comparison
```

## Files Structure

```
People_Counting/
├── app/core/
│   ├── tensorrt_quantizer.py      # Main quantization logic
│   └── tensorrt_inference.py      # TensorRT inference wrapper
├── scripts/
│   ├── quantize_models.py         # Quantization script
│   └── setup_tensorrt.py          # Installation helper
├── models/
│   ├── yolov9s.pt                # Original PyTorch
│   └── yolov9s_fp16.engine       # Quantized TensorRT
└── docs/
    └── tensorrt_guide.md          # This guide
```
