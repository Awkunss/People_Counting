# 🔍 People Counting System

Hệ thống đếm người thông minh sử dụng AI với giao diện web realtime.

## ✨ Tính năng

- 📊 **Line Counting**: Đếm người qua đường thẳng
- 🏢 **Zone Counting**: Đếm người trong vùng khu vực  
- 🌐 **Web Interface**: Giao diện web realtime với Socket.IO
- � **Interactive Drawing**: Vẽ line/zone trực tiếp trên browser
- �🎥 **Multi-source**: Hỗ trợ video file và camera
- 🤖 **AI Detection**: Sử dụng YOLO models
- 📈 **Real-time Stats**: Thống kê realtime
- 📝 **Live Logging**: Log hệ thống realtime
- 📱 **Mobile Support**: Responsive design cho mobile

## 🚀 Cài đặt và Chạy

### Phương pháp 1: Chạy Web Server (Khuyên dùng)

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Chạy web server
python main.py --method web

# Hoặc trực tiếp
python web_server.py
```

Sau đó mở browser và truy cập: `http://localhost:5000`

### Phương pháp 2: Command Line

```bash
# Line counting với video file
python main.py --method line --video Test.mp4

# Zone counting với camera
python main.py --method zone --video 0

# Hiển thị thông tin methods
python main.py --show-methods
```

## 🌐 Web Interface

### Tính năng Web UI:
- ✅ Streaming video realtime
- ✅ Điều khiển start/stop/reset
- ✅ Thống kê realtime (IN/OUT/NET/ZONE)
- ✅ Log hệ thống realtime
- ✅ Responsive design (mobile-friendly)
- ✅ Cấu hình method/video/model
- ✅ Trạng thái kết nối realtime
- ✅ **🎨 Interactive Drawing**: Vẽ line/zone bằng chuột
- ✅ **📱 Touch Support**: Vẽ bằng touch trên mobile

### Sử dụng:
1. Mở `http://localhost:5000`
2. Chọn method (Line/Zone)
3. Nhập video source (file hoặc camera ID)
4. Chọn YOLO model
5. Nhấn "Bắt đầu"
6. **🎨 Vẽ line/zone**: Click chuột để vẽ trực tiếp trên video
7. Xem real-time video và stats

## 🎨 Interactive Drawing

### **Line Counting:**
- Click **2 điểm** để tạo đường đếm
- Hệ thống hiển thị hướng IN/OUT
- Có thể vẽ lại bất cứ lúc nào

### **Zone Counting:**  
- Click **5 điểm** để tạo vùng đếm (polygon)
- Hệ thống tự động đóng polygon
- Hiển thị người trong vùng realtime

### **Controls:**
- 🎨 **Vẽ lại**: Toggle drawing mode
- 🗑️ **Xóa**: Clear drawing hiện tại
- 📱 **Mobile**: Hỗ trợ touch drawing

> 📖 **Chi tiết**: Xem `WEB_DRAWING_GUIDE.md` để biết thêm

## 📋 Requirements

```
flask==2.3.3
flask-socketio==5.3.6
opencv-python==4.8.1.78
ultralytics==8.0.200
torch>=1.13.0
torchvision>=0.14.0
numpy>=1.21.0
scipy>=1.9.0
python-socketio==5.9.0
eventlet==0.33.3
Pillow>=9.0.0
```

## 🎯 Cấu trúc Project

```
People_Counting/
├── main.py                 # Main dispatcher
├── web_server.py           # Flask web server
├── line_counting.py        # Line counting (desktop)
├── zone_counting.py        # Zone counting (desktop)
├── line_counting_web.py    # Line counting (web)
├── zone_counting_web.py    # Zone counting (web)
├── templates/
│   └── index.html          # Web interface
├── requirements.txt        # Dependencies
├── run_web_server.bat      # Windows launcher
├── run_web_server.sh       # Linux launcher
├── *.mp4                   # Video files
└── *.pt                    # YOLO models
```

## 🔧 Cấu hình

### Video Sources:
- `Test.mp4` - Video file
- `0`, `1`, `2` - Camera ID
- `rtsp://...` - RTSP stream

### YOLO Models:
- `yolov8n.pt` - Nhanh nhất
- `yolov8s.pt` - Cân bằng (mặc định)
- `yolov8m.pt` - Chính xác hơn
- `yolov8l.pt` - Chính xác nhất
- `yolov9s.pt` - YOLO v9

## 📱 Mobile Access

Web interface hỗ trợ mobile, truy cập qua:
`http://<your-computer-ip>:5000`

## 🐛 Troubleshooting

### Lỗi Camera:
- Kiểm tra camera ID (0, 1, 2...)
- Đảm bảo camera không bị ứng dụng khác sử dụng

### Lỗi Model:
- Tải models từ Ultralytics
- Kiểm tra đường dẫn file model

### Lỗi Web:
- Kiểm tra port 5000 có bị chiếm không
- Cài đặt đầy đủ dependencies

## 🎮 Keyboard Controls (Desktop mode)

- `q` - Quit
- `r` - Reset system
- `Mouse` - Click để vẽ line/zone

## 📈 Performance Tips

- Sử dụng YOLOv8n cho tốc độ cao
- Giảm resolution video nếu lag
- Sử dụng GPU nếu có CUDA

## 🤝 Contributing

1. Fork project
2. Tạo feature branch
3. Commit changes
4. Push và tạo Pull Request

## 📄 License

MIT License - Xem chi tiết trong file LICENSE System

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![YOLO](https://img.shields.io/badge/YOLO-v8-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

Hệ thống đếm người thông minh sử dụng Computer Vision và Deep Learning với hai phương pháp: **Line Counting** và **Zone Counting**.

## 🚀 Tính năng

### 📊 Line Counting
- Đếm người qua đường thẳng được định nghĩa
- Phân biệt hướng di chuyển (IN/OUT)
- Tracking chính xác với CNN features (ResNet18)
- Hiển thị trajectory của từng người

### 🏢 Zone Counting  
- Đếm số người trong vùng khu vực được định nghĩa
- Vẽ polygon tùy chỉnh (5 điểm)
- Theo dõi realtime số người trong zone
- Hiển thị lịch sử di chuyển

### 🔧 Tính năng chung
- Hỗ trợ cả video file và camera realtime
- Deep learning tracking với ResNet18 features
- YOLO object detection
- GUI tương tác với mouse
- Hiển thị FPS và thống kê realtime

## 📋 Yêu cầu hệ thống

### Dependencies
```txt
ultralytics>=8.0.0
opencv-python>=4.5.0
torch>=2.0.0
torchvision>=0.15.0
scipy>=1.9.0
numpy>=1.21.0
```

### Hardware
- **GPU**: NVIDIA GPU với CUDA support (khuyến nghị)
- **RAM**: Tối thiểu 8GB
- **Camera**: USB/IP camera (tùy chọn)

## 🛠️ Cài đặt

### 1. Clone repository
```bash
git clone https://github.com/Awkunss/People_Counting.git
cd People_Counting
```

### 2. Tạo môi trường ảo
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows
```

### 3. Cài đặt dependencies
```bash
pip install ultralytics opencv-python torch torchvision scipy numpy
```

### 4. Download YOLO model (tự động)
Model YOLO sẽ được tự động download khi chạy lần đầu.

## 🎯 Sử dụng

### Command Line Interface

#### Line Counting
```bash
# Sử dụng video file
python main.py --method line --video Test.mp4

# Sử dụng camera
python main.py --method line --video 0

# Với model khác
python main.py --method line --video Test.mp4 --model yolov8l.pt
```

#### Zone Counting
```bash
# Sử dụng video file
python main.py --method zone --video Test.mp4

# Sử dụng camera
python main.py --method zone --video 0
```

#### Hiển thị thông tin
```bash
python main.py --show-methods
```

### Chạy trực tiếp
```bash
# Line counting
python line_counting.py

# Zone counting  
python zone_counting.py
```

## 📖 Hướng dẫn sử dụng

### Line Counting
1. **Vẽ đường đếm**: Click 2 điểm trên video để tạo đường đếm
2. **Quan sát**: Hệ thống sẽ đếm người qua đường và phân biệt hướng IN/OUT
3. **Điều khiển**:
   - `Q`: Thoát
   - `R`: Reset đường đếm

### Zone Counting
1. **Vẽ vùng đếm**: Click 5 điểm để tạo polygon
2. **Quan sát**: Hệ thống hiển thị số người trong vùng realtime
3. **Điều khiển**:
   - `Q`: Thoát
   - `R`: Reset vùng đếm

## 📁 Cấu trúc dự án

```
People_Counting/
├── main.py              # Dispatcher chính
├── line_counting.py     # Line counting implementation
├── zone_counting.py     # Zone counting implementation
├── Test.mp4            # Video test mẫu
├── .gitignore          # Git ignore rules
├── README.md           # Documentation
└── requirements.txt    # Dependencies (tùy chọn)
```

## ⚙️ Tùy chỉnh

### Thay đổi model YOLO
```python
# Trong file counting
model = YOLO('yolov8s.pt')  # Small - nhanh
model = YOLO('yolov8m.pt')  # Medium  
model = YOLO('yolov8l.pt')  # Large - chính xác
model = YOLO('yolov8x.pt')  # Extra Large
```

### Điều chỉnh confidence threshold
```python
CONF_THRESHOLD = 0.5  # Giảm để detect nhiều hơn
CONF_THRESHOLD = 0.7  # Tăng để chính xác hơn
```

### Thay đổi tracking parameters
```python
tracker = DeepTracker(
    max_age=30,     # Thời gian track tối đa
    min_hits=3      # Số lần hit tối thiểu
)
```

## 📊 Performance

| Method | FPS | Accuracy | GPU Memory |
|--------|-----|----------|------------|
| Line   | 25-30 | ~95% | 2-3GB |
| Zone   | 20-25 | ~93% | 2-4GB |

*Tested on RTX 3060, 1080p video*

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

## 👨‍💻 Tác giả

**Awkunss** - [GitHub](https://github.com/Awkunss)

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)
- ResNet18 for feature extraction

## 📞 Liên hệ

- GitHub Issues: [Create an issue](https://github.com/Awkunss/People_Counting/issues)
- Email: your-email@example.com

---

⭐ **Star this repository if you find it helpful!**