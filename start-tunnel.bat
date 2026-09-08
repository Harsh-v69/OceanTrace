@echo off
REM Double-click to put OceanTrace online via ngrok.
REM Reads the ngrok token from ngrok's config and Twilio/password from .env.
REM Close the window (or press Ctrl+C) to take it offline.

cd /d "%~dp0"

REM stop any leftover server / tunnel from a previous run
taskkill /F /IM ngrok.exe >nul 2>&1
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /r /c:"127.0.0.1:8000 .*LISTENING"') do taskkill /F /PID %%p >nul 2>&1

".venv\Scripts\python.exe" scripts\deploy_live.py

echo.
echo Tunnel stopped. Press any key to close.
pause >nul
