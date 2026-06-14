@echo off
REM iRaceEngineer launcher
REM Double-click this file or schedule it to run iRaceEngineer in live mode.

call "c:\iRaceEngineer\.venv\Scripts\activate.bat"
"c:\iRaceEngineer\.venv\Scripts\python.exe" "c:\iRaceEngineer\main.py" %*
pause
