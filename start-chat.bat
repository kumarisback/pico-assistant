@echo off
title Pico Assistant - Chat Mode
cd /d "%~dp0"
echo ==============================================
echo        Pico Assistant - Text Chat
echo ==============================================
echo Connecting to ws://127.0.0.1:8080/ws ...
echo Type your message and press Enter. Type 'quit' to exit.
echo.
.\.venv\Scripts\python.exe -m clients.chat_client
pause
