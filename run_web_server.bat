@echo off
echo ========================================
echo    PEOPLE COUNTING WEB SERVER SETUP
echo ========================================

echo.
echo 📦 Installing required packages...
pip install -r requirements.txt

echo.
echo 🚀 Starting web server...
python web_server.py

pause
