@echo off
rem The native Windows on Arm64 build environment (docs/arm64.md): MSVC for Arm64 hosts and targets, the CMake and
rem Ninja that come with Visual Studio, and a CUDA toolkit (tools\port\fetch_cuda.py windows-arm64 DIR).
rem usage: call tools\port\env-win-arm64.cmd [CUDA dir]      (default: %CUDA_PATH%)
rem It puts the toolkit's bin and bin\arm64 on PATH: the tests and the engine load cuBLAS and the runtime from there.
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo env-win-arm64: vswhere.exe not found: install Visual Studio 2026 or its Build Tools with the ARM64 C++ tools
  exit /b 1
)
set "VSDIR="
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.ARM64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR (
  echo env-win-arm64: no Visual Studio with the ARM64 C++ tools
  exit /b 1
)
if not "%~1"=="" set "CUDA_PATH=%~f1"
if not defined CUDA_PATH (
  echo env-win-arm64: give the CUDA toolkit directory, or set CUDA_PATH
  exit /b 1
)
if not exist "%CUDA_PATH%\bin\nvcc.exe" (
  echo env-win-arm64: no bin\nvcc.exe under the CUDA directory %CUDA_PATH%
  exit /b 1
)
rem vcvarsall looks for vswhere on PATH
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "%VSDIR%\VC\Auxiliary\Build\vcvarsall.bat" arm64 >nul || exit /b 1
set "CUDAToolkit_ROOT=%CUDA_PATH%"
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\bin\arm64;%VSDIR%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%VSDIR%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
echo env-win-arm64: MSVC %VCToolsVersion% for Arm64, CUDA at %CUDA_PATH%
