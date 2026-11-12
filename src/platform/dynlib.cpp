// src/platform/dynlib.cpp - see include/basalt/platform/dynlib.hpp.
#include "basalt/platform/dynlib.hpp"

#include <algorithm>
#include <string>
#include <vector>

#include "basalt/platform/file.hpp"

#if defined(_WIN32)
#include "basalt/platform/win.hpp"
#else
#include <dlfcn.h>
#include <link.h>
#endif

namespace basalt::platform {

#if defined(_WIN32)
namespace {
/// A driver that keeps nvml.dll outside System32: its NVSMI directory under one of the program-file roots. The
/// root comes from the environment as UTF-16, so the path never goes through the ANSI code page. The DLL's own
/// dependencies are looked up beside it.
HMODULE load_nvsmi() {
    static const wchar_t* const roots[] = { L"ProgramFiles", L"ProgramFiles(x86)" };
    wchar_t dir[4096];
    for (const wchar_t* root : roots) {
        const DWORD n = GetEnvironmentVariableW(root, dir, (DWORD) (sizeof dir / sizeof dir[0]));
        if (n == 0 || n >= sizeof dir / sizeof dir[0]) continue;
        const std::wstring path = std::wstring(dir, n) + L"\\NVIDIA Corporation\\NVSMI\\nvml.dll";
        const HMODULE m = LoadLibraryExW(path.c_str(), nullptr,
                                         LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (m != nullptr) return m;
    }
    return nullptr;
}
}  // namespace

DynLib DynLib::open_nvml() {
    // A bare name is looked up in System32 only: the application and current directories are not searched, so
    // an nvml.dll lying next to the executable cannot stand in for the driver's.
    HMODULE m = LoadLibraryExW(L"nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (m == nullptr) m = load_nvsmi();
    DynLib lib;
    lib.h_ = (void*) m;
    return lib;
}

DynLib DynLib::open_app(const std::string& file) {
    // Each candidate by full path: LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR resolves the library's own dependencies in its
    // folder, and a bare name would never search PATH with these flags, so PATH is walked here.
    std::vector<std::wstring> dirs;
    const std::string self = module_path();
    const size_t cut = self.find_last_of("\\/");
    if (cut != std::string::npos) dirs.push_back(to_wide(self.substr(0, cut)));
    const DWORD n = GetEnvironmentVariableW(L"PATH", nullptr, 0);
    if (n > 0) {
        std::wstring path((size_t) n, L'\0');
        const DWORD got = GetEnvironmentVariableW(L"PATH", &path[0], n);
        path.resize(got < n ? got : 0);
        size_t at = 0;
        while (at <= path.size()) {
            const size_t end = std::min(path.find(L';', at), path.size());
            std::wstring d = path.substr(at, end - at);
            if (d.size() >= 2 && d.front() == L'"' && d.back() == L'"') d = d.substr(1, d.size() - 2);
            // only absolute entries (a drive or a share): a relative one would name the current directory
            const bool absolute = (d.size() >= 3 && d[1] == L':' && (d[2] == L'\\' || d[2] == L'/')) ||
                                  d.rfind(L"\\\\", 0) == 0;
            if (absolute) dirs.push_back(d);
            at = end + 1;
        }
    }
    const std::wstring name = to_wide(file);
    DynLib lib;
    for (const std::wstring& d : dirs) {
        const std::wstring full = d + (d.back() == L'\\' || d.back() == L'/' ? L"" : L"\\") + name;
        const HMODULE m = LoadLibraryExW(full.c_str(), nullptr,
                                         LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (m != nullptr) { lib.h_ = (void*) m; break; }
    }
    return lib;
}

std::string DynLib::versioned_name(const char* name, int major) { return std::string(name) + "-" + std::to_string(major) + ".dll"; }

std::string DynLib::path() const {
    if (h_ == nullptr) return std::string();
    wchar_t buf[4096];
    const DWORD cap = (DWORD) (sizeof buf / sizeof buf[0]) - 1;
    const DWORD n = GetModuleFileNameW((HMODULE) h_, buf, cap);
    if (n == 0 || n >= cap) return std::string();
    buf[n] = 0;
    return to_utf8(buf);
}

void* DynLib::address(const char* name) const {
    return h_ == nullptr ? nullptr : (void*) GetProcAddress((HMODULE) h_, name);
}

void DynLib::reset() {
    // The library stays loaded for the life of the process: the function pointers handed out stay valid, and the
    // management library keeps threads of its own that must not outlive their code.
    h_ = nullptr;
}
#else
DynLib DynLib::open_nvml() {
    DynLib lib;
    lib.h_ = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    return lib;
}

DynLib DynLib::open_app(const std::string& file) {
    DynLib lib;
    lib.h_ = dlopen(file.c_str(), RTLD_NOW | RTLD_LOCAL);
    return lib;
}

std::string DynLib::versioned_name(const char* name, int major) { return "lib" + std::string(name) + ".so." + std::to_string(major); }

std::string DynLib::path() const {
    struct link_map* lm = nullptr;
    if (h_ == nullptr || dlinfo(h_, RTLD_DI_LINKMAP, &lm) != 0 || lm == nullptr || lm->l_name == nullptr) return std::string();
    return lm->l_name;
}

void* DynLib::address(const char* name) const { return h_ == nullptr ? nullptr : dlsym(h_, name); }

void DynLib::reset() { h_ = nullptr; }   // never dlclosed, as before: the management library stays loaded
#endif

}  // namespace basalt::platform
