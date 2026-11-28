# Platform code

OS code lives only in `src/platform/` and `include/basalt/platform/`, and CMake `if(WIN32)` blocks. Anywhere else there is no `_WIN32`,
`_MSC_VER`, `__linux__`, `<windows.h>`, `<sys/mman.h>`, `<unistd.h>` or raw OS call (mmap, madvise, pread, pthread/affinity,
fork/exec/pipe, dlopen, /proc or /sys reads). `tools/check_platform_boundary.sh` enforces it. Each seam has a Linux and a Windows body,
the Linux body being the code you would have written anyway, and a test in `tests/platform/` that runs on both. Paths are UTF-8 and go
through the platform file calls.
