@echo off
title Project 8 - Local Demo Launcher
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv\Scripts\python.exe not found!
    echo Please create the virtual environment and install requirements.txt first.
    pause
    exit /b 1
)

echo ============================================================================
echo   PROJECT 8: WEARABLE MOVEMENT RECOGNITION - ONE-CLICK LOCAL DEMO
echo ============================================================================
echo.
echo Starting both services in separate windows...
echo   1. Live Phone Receiver & Model Server (Port 8765)
echo   2. Streamlit Benchmark & Check Dashboard (Port 8501)
echo.

start "Live Phone Server (Port 8765)" cmd /k "".venv\Scripts\python.exe" -m src.live_server"
timeout /t 2 /nobreak >nul
start "Streamlit Dashboard (Port 8501)" cmd /k "".venv\Scripts\python.exe" -m streamlit run app/dashboard.py"

echo ============================================================================
echo   HOW TO SHOW THE DEMO TO SIR:
echo ============================================================================
echo   1. Keep laptop and Android phone on the SAME Wi-Fi (or phone hotspot).
echo   2. Check the "Live Phone Server" window for your Wi-Fi URL, e.g.:
echo        http://192.168.x.x:8765
echo   3. On Android Chrome, open:
echo        chrome://flags/#unsafely-treat-insecure-origin-as-secure
echo      - Set to ENABLED, paste the exact http://192.168.x.x:8765 URL,
echo      - Tap RELAUNCH.
echo   4. Open http://192.168.x.x:8765 on the phone.
echo   5. Select a movement label, tap "Allow motion access and start recording",
echo      move for 3+ seconds, then tap "Stop sensing and save recording".
echo   6. On the laptop Streamlit dashboard (http://localhost:8501), open
echo      "Check my phone recording" to compare live predictions vs labels!
echo.
echo   [TIP] If college Wi-Fi blocks device-to-device traffic, turn on your
echo         phone's Mobile Hotspot and connect the laptop to it!
echo ============================================================================
echo.
pause
