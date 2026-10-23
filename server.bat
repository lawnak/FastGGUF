@echo off
rem Drag one or more .gguf files onto this file to start the web UI + OpenAI-compatible API.
rem Or run:  server.bat model.gguf [model2.gguf ...]
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  echo Drag one or more .gguf model files onto server.bat, or run: server.bat model.gguf
  pause
  exit /b 1
)
if "%PORT%"=="" set PORT=8080
rem open the browser a few seconds later, once the model has loaded
start "" /min cmd /c "timeout /t 8 >nul & start http://localhost:%PORT%/"
fastgguf.exe serve %* %PORT% 4096
pause
