@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin;%PATH%"
cd /d D:\test\basalt-fork
"C:\Program Files\Microsoft Visual Studio\18\Insiders\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build-fork --target basalt -j 6
exit /b %ERRORLEVEL%
