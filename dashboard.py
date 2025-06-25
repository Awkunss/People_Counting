#!/usr/bin/env python3
"""
People Counter Dashboard
========================
Dashboard Streamlit để hiển thị dữ liệu People Counter
"""

import streamlit as st
import pandas as pd
import json


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
        st.error(f"❌ Không tìm thấy file '{filename}'. Vui lòng chạy 'python generate_data.py' trước.")
        return None


def main():
    """
    Hàm chính của dashboard
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
        
        # Thêm sidebar để tùy chỉnh
        st.sidebar.markdown("### ⚙️ Tùy chỉnh")
        st.sidebar.markdown("**Lọc dữ liệu:**")
        
        # Lọc theo thời gian
        time_range = st.sidebar.slider(
            "Chọn khoảng thời gian",
            min_value=0,
            max_value=len(df)-1,
            value=(0, len(df)-1)
        )
        
        filtered_df = df.iloc[time_range[0]:time_range[1]+1]
        
        if st.sidebar.button("🔄 Cập nhật biểu đồ"):
            st.line_chart(filtered_df.set_index("timestamp")["count"])
    
    # Footer
    st.markdown(
        "<hr><center><i>Dashboard demo mô phỏng theo slide LAB01_DAT - People Counting</i></center>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main() 