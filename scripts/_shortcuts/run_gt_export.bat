@echo off
cd /d "%~dp0\..\.."
echo [*] Running Hybrid ByteTrack with CVAT XML export (100 frames)...
.venv\Scripts\python.exe scripts\track_videos.py -v Pigs281119_000085_30fps --max-frames 100 --mode hybrid_bytetrack
echo.
echo [OK] Execution finished.
pause
