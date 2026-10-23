@echo off
rem Build fastgguf on Windows with gcc from w64devkit or MSYS2 (UCRT64 / MINGW64).
rem   build.bat            optimized for this PC
rem   build.bat portable   runs on any PC with an AVX2 CPU
setlocal
where gcc >nul 2>nul || (
  echo gcc not found. Install w64devkit ^(https://github.com/skeeto/w64devkit^) or MSYS2 and add it to PATH.
  exit /b 1
)
set ARCH=-march=native
if /i "%1"=="portable" set ARCH=-march=x86-64-v3
set FLAGS=-O3 %ARCH% -fopenmp -static
echo [1/3] fastgguf.exe
gcc %FLAGS% -o fastgguf.exe fastgguf.c -lm -lws2_32 || exit /b 1
echo [2/3] fastgguf-server.exe
gcc %FLAGS% -DFG_SERVER_MAIN -o fastgguf-server.exe fastgguf.c -lm -lws2_32 || exit /b 1
echo [3/3] fastgguf-console.exe
gcc %FLAGS% -DFG_CONSOLE_MAIN -o fastgguf-console.exe fastgguf.c -lm -lws2_32 || exit /b 1
echo.
echo Done. Test it with:  fastgguf.exe selftest
