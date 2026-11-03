// include/basalt/platform/dynlib.hpp - a shared library loaded at run time, for the libraries the engine reads
// optionally (the GPU management library, FFmpeg's): a machine without one still runs, and the caller says so.
#pragma once

#include <string>

namespace basalt::platform {

class DynLib {
public:
    DynLib() = default;
    DynLib(DynLib&& o) noexcept : h_(o.h_) { o.h_ = nullptr; }
    DynLib& operator=(DynLib&& o) noexcept {
        if (this != &o) { reset(); h_ = o.h_; o.h_ = nullptr; }
        return *this;
    }
    DynLib(const DynLib&) = delete;
    DynLib& operator=(const DynLib&) = delete;
    ~DynLib() { reset(); }

    /// The NVIDIA management library (NVML) of the installed driver; empty when there is none. Where the loader
    /// searches is the platform's own, and never the application's directory, so a library beside the executable
    /// cannot stand in for the driver's.
    static DynLib open_nvml();

    /// An application library by its file name (`versioned_name`). Linux: the system loader's own search (the
    /// library path, its cache, the default directories). Windows: the executable's folder, then each PATH entry,
    /// each tried by full path so the library's own dependencies resolve beside it; never the current directory.
    /// Empty when no such library loads.
    static DynLib open_app(const std::string& file);

    /// The file name a library built with its major version in the name has here: `libNAME.so.MAJOR` on Linux,
    /// `NAME-MAJOR.dll` on Windows (FFmpeg's names on both).
    static std::string versioned_name(const char* name, int major);

    /// The file the library was loaded from; empty when the platform does not say.
    std::string path() const;

    explicit operator bool() const { return h_ != nullptr; }

    /// The function `name`, or null when the library has no such symbol.
    template <class F>
    F sym(const char* name) const { return reinterpret_cast<F>(address(name)); }

    void reset();

private:
    void* address(const char* name) const;
    void* h_ = nullptr;
};

}  // namespace basalt::platform
