@echo off
title Pico Assistant - Voice Mode
cd /d "%~dp0"
echo ========================================================
echo             Pico Assistant - Voice Client
echo ========================================================
echo Connecting to ws://127.0.0.1:8080/ws ...
echo Watch the [mic: ...] volume bar to ensure your mic is active!
echo Press Ctrl+C to quit.
echo ========================================================
echo.
.\.venv\Scripts\python.exe -m clients.mic_client --gain 15.0
pause
