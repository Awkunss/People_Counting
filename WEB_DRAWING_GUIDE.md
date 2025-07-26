# 🎨 Hướng dẫn sử dụng tính năng vẽ trên Web Interface

## ✨ Tính năng mới: Vẽ Line và Zone trực tiếp trên trình duyệt

Giờ đây bạn có thể tùy chỉnh đường đếm (Line) và vùng đếm (Zone) trực tiếp trên trình duyệt, tương tự như phiên bản desktop!

## 🚀 Cách sử dụng

### 1. **Khởi động hệ thống**
```bash
python main.py --method web
# hoặc
python web_server.py
```

### 2. **Truy cập Web Interface**
- Mở browser: `http://localhost:5000`
- Chọn method (Line hoặc Zone)
- Chọn video source và model
- Nhấn "🚀 Bắt đầu"

### 3. **Chế độ vẽ tự động bật**
Khi hệ thống chạy, chế độ vẽ sẽ tự động bật với:
- 📍 Hướng dẫn vẽ ở góc trên
- 🎯 Con trỏ chuyển thành crosshair
- 📝 Canvas overlay trên video

## 🎯 Vẽ Line Counting

### **Cách vẽ:**
1. **Click 2 điểm** trên video để tạo đường đếm
2. Điểm 1: Điểm bắt đầu
3. Điểm 2: Điểm kết thúc
4. Hệ thống tự động hoàn thành sau 2 điểm

### **Tính năng:**
- ✅ Đường màu đỏ hiển thị line counting
- ✅ Điểm xanh lá đánh số thứ tự
- ✅ Hiển thị hướng IN/OUT
- ✅ Realtime tracking qua đường

### **Controls:**
- 🗑️ **Xóa**: Xóa đường hiện tại và vẽ lại
- 🎨 **Vẽ lại**: Bật/tắt chế độ vẽ

## 🏢 Vẽ Zone Counting

### **Cách vẽ:**
1. **Click 5 điểm** để tạo polygon (tối thiểu 3 điểm)
2. Các điểm sẽ được nối thành đa giác
3. Hệ thống tự động đóng polygon

### **Tính năng:**
- ✅ Polygon màu xanh lá
- ✅ Fill semi-transparent
- ✅ Điểm đánh số thứ tự
- ✅ Đếm người trong vùng realtime
- ✅ Hiển thị trajectory

### **Controls:**
- 🗑️ **Xóa**: Xóa vùng hiện tại và vẽ lại
- 🎨 **Vẽ lại**: Bật/tắt chế độ vẽ

## 🎮 Controls và Hotkeys

### **Buttons:**
- 🚀 **Bắt đầu**: Khởi động counting + auto-enable drawing
- ⏹️ **Dừng**: Dừng counting + disable drawing  
- 🔄 **Reset**: Reset stats + clear drawing
- 🎨 **Vẽ lại**: Toggle drawing mode
- 🗑️ **Xóa**: Clear current drawing

### **Mouse Interaction:**
- **Left Click**: Thêm điểm vẽ
- **Crosshair cursor**: Khi trong drawing mode

## 🔧 Technical Details

### **Canvas Overlay:**
- Canvas được overlay trên video stream
- Coordinate mapping chính xác với video frame
- Real-time redraw với mỗi frame

### **Socket Events:**
- `set_drawing_points`: Gửi points tới server
- `clear_drawing`: Xóa drawing trên server
- Auto-sync giữa frontend và backend

### **Auto vs Manual:**
- **Auto**: Hệ thống tự tạo line/zone mặc định
- **Manual**: User vẽ => ghi đè auto setup
- **Flag**: `user_defined_line/zone` để track trạng thái

## 📱 Mobile Support

### **Responsive Design:**
- ✅ Hoạt động trên tablet và phone
- ✅ Touch events cho mobile
- ✅ UI scale phù hợp màn hình nhỏ

### **Touch Interaction:**
- **Tap**: Thêm điểm (tương đương click)
- **Pinch/Zoom**: Browser native zoom

## 🎨 Visual Feedback

### **Line Mode:**
```
🔴 Red Line: Counting line
🟢 Green Dots: Control points  
🟡 Yellow Numbers: Point order
📊 IN/OUT Labels: Direction indicators
```

### **Zone Mode:**
```
🟢 Green Polygon: Counting zone
🟢 Green Fill: Semi-transparent area
🟢 Green Dots: Control points
🟡 Yellow Numbers: Point order
🔵 Blue Tracks: Person trajectories
```

## 🐛 Troubleshooting

### **Drawing không hoạt động:**
- ✅ Kiểm tra hệ thống đã "Bắt đầu" chưa
- ✅ Refresh page và thử lại
- ✅ Kiểm tra console browser (F12)

### **Points không chính xác:**
- ✅ Đảm bảo video đã load hoàn toàn
- ✅ Click chậm và chính xác
- ✅ Tránh click khi video đang buffer

### **Canvas không hiển thị:**
- ✅ Browser hỗ trợ HTML5 Canvas
- ✅ JavaScript enabled
- ✅ Không bị ad-blocker chặn

## 💡 Tips và Best Practices

### **Line Drawing:**
- 🎯 Vẽ đường vuông góc với hướng di chuyển
- 📏 Đường không quá ngắn hoặc quá dài
- 🔄 Test với một vài người để verify

### **Zone Drawing:**
- 📐 Tạo polygon đơn giản, không self-intersect
- 🏢 Vùng không quá nhỏ hoặc quá lớn
- 👥 Đảm bảo cover khu vực quan tâm

### **Performance:**
- ⚡ Vẽ ít điểm hơn = performance tốt hơn
- 📺 Video resolution thấp = drawing responsive hơn
- 🖥️ Desktop browser = experience tốt nhất

## 🆕 So sánh với Desktop Version

| Tính năng | Desktop | Web |
|-----------|---------|-----|
| Mouse Click | ✅ | ✅ |
| Visual Feedback | ✅ | ✅ |
| Real-time Drawing | ✅ | ✅ |
| Multiple Points | ✅ | ✅ |
| Reset/Clear | ✅ | ✅ |
| Auto Setup | ✅ | ✅ |
| Mobile Support | ❌ | ✅ |
| Remote Access | ❌ | ✅ |
| Multi-user | ❌ | ✅ |

## 🎉 Kết luận

Tính năng vẽ trên web interface mang lại:
- 🌐 **Accessibility**: Truy cập từ mọi thiết bị
- 🎨 **Flexibility**: Tùy chỉnh dễ dàng như desktop
- 📱 **Mobile-friendly**: Hoạt động trên mobile
- 🔄 **Real-time**: Sync tức thì với backend
- 👥 **Multi-user**: Nhiều người có thể truy cập

Perfect cho việc setup và monitor hệ thống đếm người từ xa!
