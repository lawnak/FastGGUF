@echo off
rem Drag a .gguf file onto this file to start chatting in the terminal.
rem Or run:  chat.bat path\to\model.gguf [ctx]
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  echo Drag a .gguf model file onto chat.bat, or run: chat.bat model.gguf [ctx]
  pause
  exit /b 1
)
set CTX=%~2
if "%CTX%"=="" set CTX=4096
fastgguf.exe chat "%~1" %CTX%
pause
