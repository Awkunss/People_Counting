import streamlit as st
from main import realtime_log

st.title("Cập nhật số người và thời gian")

if st.button("Lấy số người hiện tại"):
    log = realtime_log()
    st.markdown(f"""
    ### Thông tin mới nhất:
    - ⏰ <b>Thời gian:</b> <span style="color:blue">{log['time']}</span>
    - 👤 <b>Số người:</b> <span style="color:green">{log['number']}</span>
    """, unsafe_allow_html=True)
