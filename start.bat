@echo off
rem SAGE demo launcher (Windows). Double-click this file.
rem Tries: py launcher -> python -> npx serve. Opens your default browser. Ctrl+C to stop.
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
  echo Starting with the Python launcher "py"...
  py -3 tools\serve.py
  if not errorlevel 9009 if not errorlevel 1 goto :done
)

where python >nul 2>nul
if %errorlevel%==0 (
  echo Starting with "python"...
  python tools\serve.py
  if not errorlevel 9009 if not errorlevel 1 goto :done
)

where npx >nul 2>nul
if %errorlevel%==0 (
  echo Python not usable; trying "npx serve" ^(needs internet the first time^)...
  start "" "http://localhost:8000/index.html"
  npx --yes serve -l 8000 .
  goto :done
)

echo.
echo ============================================================
echo  Could not start a local server: no Python 3 and no Node.js.
echo  Options:
echo    1. Install Python 3 from https://www.python.org/downloads/
echo       (tick "Add python.exe to PATH") and run start.bat again.
echo    2. No install needed: double-click  dist\sage-demo.html
echo       (single self-contained file, works offline).
echo ============================================================
pause
exit /b 1

:done
endlocal
