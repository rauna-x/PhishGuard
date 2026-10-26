@echo off
echo ============================================
echo   PhishGuard - Phishing Detection Server
echo ============================================
echo.
cd /d "%~dp0"
echo Starting Flask server on http://localhost:5000
echo Press Ctrl+C to stop.
echo.
python app.py
pause

