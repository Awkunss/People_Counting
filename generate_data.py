#!/usr/bin/env python3
"""
Generate Simulation Data
========================
Tạo dữ liệu mô phỏng cho People Counter System
"""

import random
import json
from datetime import datetime, timedelta


def generate_simulation_data(duration_minutes=30, interval_seconds=10):
    """
    Tạo dữ liệu mô phỏng về số lượng người thay đổi theo thời gian
    
    Args:
        duration_minutes (int): Thời gian mô phỏng (phút)
        interval_seconds (int): Khoảng thời gian giữa các cập nhật (giây)
    
    Returns:
        list: Danh sách các thay đổi số người
    """
    num_entries = duration_minutes * 60 // interval_seconds
    log_data = []
    current_people = 0
    start_time = datetime.now()
    
    print(f"🔄 Đang tạo dữ liệu mô phỏng trong {duration_minutes} phút...")
    
    for i in range(num_entries):
        # Tạo timestamp cho mỗi cập nhật
        timestamp = start_time + timedelta(seconds=i * interval_seconds)
        
        # Tạo thay đổi ngẫu nhiên (-1, 0, +1 người)
        change = random.choice([-1, 0, 1])
        new_people = max(0, current_people + change)  # Đảm bảo không âm
        
        # Chỉ ghi log khi có thay đổi
        if new_people != current_people:
            log_data.append({
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "count": new_people
            })
            current_people = new_people
    
    return log_data


def save_data_to_json(data, filename="people_log.json"):
    """
    Lưu dữ liệu vào file JSON
    
    Args:
        data (list): Dữ liệu cần lưu
        filename (str): Tên file
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"✅ Đã lưu {len(data)} bản ghi vào '{filename}'")


def main():
    """
    Hàm chính để tạo dữ liệu
    """
    print("🚀 People Counter - Data Generator")
    print("=" * 40)
    
    # Tạo dữ liệu mô phỏng
    print("\n1️⃣ Tạo dữ liệu mô phỏng...")
    simulation_data = generate_simulation_data(duration_minutes=30, interval_seconds=10)
    save_data_to_json(simulation_data)
    
    # Hiển thị thông tin
    print(f"\n📊 Thống kê dữ liệu:")
    print(f"- Tổng số thay đổi: {len(simulation_data)}")
    print(f"- Số người cuối cùng: {simulation_data[-1]['count'] if simulation_data else 0}")
    print(f"- Thời gian mô phỏng: 30 phút")
    print(f"- Khoảng thời gian cập nhật: 10 giây")
    
    print("\n✅ Hoàn thành! Bạn có thể chạy dashboard bằng lệnh:")
    print("streamlit run dashboard.py")


if __name__ == "__main__":
    main() 