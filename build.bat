@echo off
rem Build fastgguf on Windows with gcc from w64devkit or MSYS2 (UCRT64 / MINGW64).
rem   build.bat            optimized for this PC
rem   build.bat portable   runs on any 64-bit PC: each .exe carries AVX-512 VNNI, AVX-VNNI, AVX2 and
rem                        plain x86-64 builds of the engine and starts the fastest one the CPU supports
setlocal
where gcc >nul 2>nul || (
  echo gcc not found. Install w64devkit ^(https://github.com/skeeto/w64devkit^) or MSYS2 and add it to PATH.
  exit /b 1
)
set LIBS=-lm -lws2_32
rem MinGW gcc does not always keep the stack aligned for AVX (gcc bug 54412): use unaligned vector moves
set XF=
echo int main(void){return 0;}> "%TEMP%\fg_probe.c"
gcc -Wa,-muse-unaligned-vector-move -c -o "%TEMP%\fg_probe.o" "%TEMP%\fg_probe.c" >nul 2>nul && set XF=-Wa,-muse-unaligned-vector-move
rem fully static .exe when the toolchain has a static OpenMP library; MSYS2 may only have the DLL
set LINK=-static
gcc -fopenmp -static -o "%TEMP%\fg_probe.exe" "%TEMP%\fg_probe.c" >nul 2>nul || set LINK=-static-libgcc
if "%LINK%"=="-static-libgcc" echo static OpenMP not available - linking dynamically ^(keep libgomp-1.dll next to the .exe^)
set CF=-O3 -fopenmp %XF%
if /i "%1"=="portable" goto portable

echo [1/3] fastgguf.exe
gcc %CF% -march=native -o fastgguf.exe fastgguf.c %LINK% %LIBS% || exit /b 1
echo [2/3] fastgguf-server.exe
gcc %CF% -march=native -DFG_SERVER_MAIN -o fastgguf-server.exe fastgguf.c %LINK% %LIBS% || exit /b 1
echo [3/3] fastgguf-console.exe
gcc %CF% -march=native -DFG_CONSOLE_MAIN -o fastgguf-console.exe fastgguf.c %LINK% %LIBS% || exit /b 1
goto done

:portable
echo [1/7] engine, AVX-512 VNNI
gcc %CF% -march=x86-64-v4 -mavx512vnni -DFG_VARIANT=avx512 -c -o port-avx512.o fastgguf.c || exit /b 1
echo [2/7] engine, AVX-VNNI
gcc %CF% -march=x86-64-v3 -mavxvnni -DFG_VARIANT=avxvnni -c -o port-avxvnni.o fastgguf.c || exit /b 1
echo [3/7] engine, AVX2
gcc %CF% -march=x86-64-v3 -DFG_VARIANT=avx2 -c -o port-avx2.o fastgguf.c || exit /b 1
echo [4/7] engine, plain x86-64
gcc %CF% -march=x86-64 -DFG_VARIANT=sse2 -c -o port-sse2.o fastgguf.c || exit /b 1
set OBJS=port-avx512.o port-avxvnni.o port-avx2.o port-sse2.o
echo [5/7] fastgguf.exe
gcc %CF% -o fastgguf.exe dispatch.c %OBJS% %LINK% %LIBS% || exit /b 1
echo [6/7] fastgguf-server.exe
gcc %CF% -DFG_SERVER_MAIN -o fastgguf-server.exe dispatch.c %OBJS% %LINK% %LIBS% || exit /b 1
echo [7/7] fastgguf-console.exe
gcc %CF% -DFG_CONSOLE_MAIN -o fastgguf-console.exe dispatch.c %OBJS% %LINK% %LIBS% || exit /b 1
del %OBJS%

:done
echo.
echo Done. Test it with:  fastgguf.exe selftest
