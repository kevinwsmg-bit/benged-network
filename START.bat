@echo off
title Benged Network - keep this window open while you stream
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo  Run SETUP.bat first.
  pause
  exit /b 1
)
start "" http://localhost:8765/
rem launcher.py installs updates from GitHub, then runs the overlay (and restarts it after an update)
.venv\Scripts\python launcher.py
echo.
echo  The overlay stopped. Screenshot this window and send it to Kevin.
pause
