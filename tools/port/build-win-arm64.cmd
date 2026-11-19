@echo off
rem Configures (on the first run) and builds Basalt natively on Windows on Arm64 (docs/arm64.md).
rem usage: tools\port\build-win-arm64.cmd [build dir] [CUDA dir]      (defaults: build, %CUDA_PATH%)
rem The binaries land in bin\ of the source tree. Tests: call tools\port\env-win-arm64.cmd, then ctest --test-dir BUILD.
setlocal
for %%i in ("%~dp0..\..") do set "SRC=%%~fi"
set "BUILD=%~1"
if "%BUILD%"=="" set "BUILD=%SRC%\build"
call "%~dp0env-win-arm64.cmd" %2 || exit /b 1
if not exist "%BUILD%\build.ninja" (
  cmake -S "%SRC%" -B "%BUILD%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON || exit /b 1
)
cmake --build "%BUILD%" || exit /b 1
echo build-win-arm64: built; binaries in %SRC%\bin
