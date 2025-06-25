#!/usr/bin/env python3
"""
People Counter System
=====================
Hệ thống mô phỏng đếm người với dashboard Streamlit
Từ LAB01_DAT - People Counting
"""

import random
import json
import pandas as pd
import streamlit as st
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


def load_data_from_json(filename="people_log.json"):
    """
    Đọc dữ liệu từ file JSON
    
    Args:
        filename (str): Tên file
    
    Returns:
        pandas.DataFrame: DataFrame chứa dữ liệu
    """
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file '{filename}'")
        return None


def create_streamlit_dashboard():
    """
    Tạo dashboard Streamlit để hiển thị dữ liệu
    """
    st.set_page_config(
        page_title="People Counter Dashboard",
        layout="centered",
        page_icon="👥"
    )
    
    # Tiêu đề
    st.markdown(
        "<h1 style='text-align: center; color: #4CAF50;'>📊 People Counting Dashboard</h1>",
        unsafe_allow_html=True
    )
    
    # Tải dữ liệu
    df = load_data_from_json()
    
    if df is not None:
        # Thống kê nhanh
        latest_count = df["count"].iloc[-1]
        total_changes = len(df)
        max_count = df["count"].max()
        min_count = df["count"].min()
        
        st.markdown("### 👥 Thống kê nhanh")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Số người hiện tại", latest_count)
        with col2:
            st.metric("Số lần thay đổi", total_changes)
        with col3:
            st.metric("Số người cao nhất", max_count)
        with col4:
            st.metric("Số người thấp nhất", min_count)
        
        # Biểu đồ
        st.markdown("### 📈 Biểu đồ số người theo thời gian")
        st.line_chart(df.set_index("timestamp")["count"])
        
        # Bảng dữ liệu
        st.markdown("### 📋 Bảng log chi tiết")
        st.dataframe(df, use_container_width=True)
        
        # Thống kê bổ sung
        st.markdown("### 📊 Phân tích chi tiết")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Thời gian hoạt động:**")
            st.write(f"- Bắt đầu: {df['timestamp'].min().strftime('%H:%M:%S')}")
            st.write(f"- Kết thúc: {df['timestamp'].max().strftime('%H:%M:%S')}")
            st.write(f"- Tổng thời gian: {(df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 60:.1f} phút")
        
        with col2:
            st.write("**Thống kê số người:**")
            st.write(f"- Trung bình: {df['count'].mean():.1f}")
            st.write(f"- Độ lệch chuẩn: {df['count'].std():.1f}")
            st.write(f"- Phạm vi: {min_count} - {max_count}")
        
    else:
        st.error("❌ Không tìm thấy file 'people_log.json'. Vui lòng tạo dữ liệu trước.")
    
    # Footer
    st.markdown(
        "<hr><center><i>Dashboard demo mô phỏng theo slide LAB01_DAT - People Counting</i></center>",
        unsafe_allow_html=True
    )


def main():
    """
    Hàm chính để chạy hệ thống
    """
    print("🚀 People Counter System")
    print("=" * 50)
    
    # Tạo dữ liệu mô phỏng
    print("\n1️⃣ Tạo dữ liệu mô phỏng...")
    simulation_data = generate_simulation_data(duration_minutes=30, interval_seconds=10)
    save_data_to_json(simulation_data)
    
    # Hiển thị thông tin
    print(f"\n📊 Thống kê dữ liệu:")
    print(f"- Tổng số thay đổi: {len(simulation_data)}")
    print(f"- Số người cuối cùng: {simulation_data[-1]['count'] if simulation_data else 0}")
    print(f"- Thời gian mô phỏng: 30 phút")
    
    print("\n2️⃣ Khởi động dashboard...")
    print("🌐 Mở trình duyệt và truy cập: http://localhost:8501")
    print("💡 Nhấn Ctrl+C để dừng server")
    
    # Chạy Streamlit dashboard
    import subprocess
    import sys
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__, "--server.headless", "true"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Đã dừng server dashboard")
    except Exception as e:
        print(f"❌ Lỗi khi chạy dashboard: {e}")


if __name__ == "__main__":
    # Kiểm tra xem có phải đang chạy Streamlit không
    try:
        # Nếu đang chạy trong Streamlit
        create_streamlit_dashboard()
    except:
        # Nếu chạy trực tiếp
        main() 