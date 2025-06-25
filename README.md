# People Counter System

Hệ thống mô phỏng đếm người với dashboard Streamlit - Từ LAB01_DAT - People Counting

## 📋 Mô tả

Hệ thống này mô phỏng việc đếm số lượng người thay đổi theo thời gian và hiển thị kết quả qua dashboard web.

## 🚀 Cách sử dụng

### 1. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 2. Tạo dữ liệu mô phỏng
```bash
python generate_data.py
```

### 3. Chạy dashboard
```bash
streamlit run dashboard.py
```

### 4. Truy cập dashboard
Mở trình duyệt và truy cập: http://localhost:8501

## 📊 Tính năng

### Tạo dữ liệu mô phỏng (`generate_data.py`)
- Mô phỏng trong 30 phút
- Cập nhật mỗi 10 giây
- Số người có thể tăng (+1), giảm (-1) hoặc không đổi (0)
- Lưu dữ liệu vào file `people_log.json`

### Dashboard Streamlit (`dashboard.py`)
- **Thống kê nhanh**: Số người hiện tại, số lần thay đổi, min/max
- **Biểu đồ**: Đường thời gian số người
- **Bảng dữ liệu**: Log chi tiết các thay đổi
- **Phân tích**: Thời gian hoạt động, thống kê chi tiết
- **Tùy chỉnh**: Lọc dữ liệu theo thời gian

## 📁 Cấu trúc file

```
hai/
├── generate_data.py           # Tạo dữ liệu mô phỏng
├── dashboard.py              # Dashboard Streamlit
├── people_counter_system.py  # File tổng hợp (cũ)
├── requirements.txt          # Dependencies
├── README.md                 # Hướng dẫn
├── people_log.json          # Dữ liệu mô phỏng (tự tạo)
└── Untitled (1).ipynb       # Notebook gốc
```

## 🔧 Tùy chỉnh

### Thay đổi thông số mô phỏng
Trong file `generate_data.py`, bạn có thể thay đổi:
- `duration_minutes`: Thời gian mô phỏng (mặc định: 30 phút)
- `interval_seconds`: Khoảng thời gian cập nhật (mặc định: 10 giây)

### Thêm tính năng dashboard
Trong file `dashboard.py`, bạn có thể:
- Thêm biểu đồ mới
- Tùy chỉnh giao diện
- Thêm tính năng lọc dữ liệu

## 📝 Lưu ý

- File `people_log.json` sẽ được tạo tự động khi chạy `generate_data.py`
- Dashboard sẽ tự động reload khi có thay đổi dữ liệu
- Nhấn Ctrl+C để dừng server Streamlit
- Có thể chạy lại `generate_data.py` để tạo dữ liệu mới

## 🎯 Ví dụ sử dụng

```bash
# Bước 1: Tạo dữ liệu
python generate_data.py

# Bước 2: Chạy dashboard
streamlit run dashboard.py

# Bước 3: Mở trình duyệt
# Truy cập: http://localhost:8501
``` 