#include "serve.hpp"

#include "basalt/log.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/process.hpp"
#include "basalt/platform/system.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstring>

namespace basalt::program::cli {

/// `basalt serve --config FILE` or `basalt serve MODEL.basalt [--python PY] [server args...]`: runs the Python server
/// over this binary.  The server gets `--engine-bin` and what was typed; a MODEL.basalt argument stands for `--model MODEL.basalt`.
/// Only returns on failure.
int serve_subcommand(int argc, char** argv) {
    std::string python;
    std::vector<std::string> extra;
    bool have_model = false;
    // the working directory changes below: the paths the server reads must not depend on it
    auto absolute = [](const std::string& path) {
        std::string real;
        if (!basalt::platform::absolute_path(path, real)) {
            basalt::log::error("basalt serve", "cannot open %s", path.c_str());
            std::exit(1);
        }
        return real;
    };
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) { basalt::log::error("basalt serve", "%s needs a value", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--python") python = value();
        else if (a == "--model") { extra.insert(extra.end(), {a, absolute(value())}); have_model = true; }
        else if (a == "--config") extra.insert(extra.end(), {a, absolute(value())});
        // only the first argument can be MODEL.basalt; a later bare word is the value of the option before it
        else if (i == 2 && !have_model && !a.empty() && a[0] != '-') { extra.insert(extra.end(), {"--model", absolute(a)}); have_model = true; }
        else extra.push_back(a);
    }
    if (extra.empty()) {
        basalt::log::error("basalt", "usage: basalt serve --config FILE | MODEL.basalt [--python PY] [server args...]");
        return 2;
    }
    const std::string exe = basalt::platform::module_path();
    if (exe.empty()) { basalt::log::error("basalt serve", "cannot find this binary"); return 1; }
    const std::string src_dir = BASALT_SOURCE_DIR;
    if (python.empty()) python = basalt::platform::python_interpreter(src_dir);
    if (basalt::platform::set_working_dir(src_dir) != 0) {
        basalt::log::error("basalt serve", "cannot enter the source tree %s", src_dir.c_str());
        return 1;
    }
    std::vector<std::string> args = {python, "-m", "serve.server", "--engine-bin", exe};
    args.insert(args.end(), extra.begin(), extra.end());
    // Does not return when it works. Windows has no exec, so this process waits for the server and ends with the
    // server's exit code. Reaching the line below means the server never started, and errno says why.
    basalt::platform::exec_replacing(args);
    basalt::log::error("basalt serve", "cannot run %s: %s", python.c_str(), std::strerror(errno));
    return 1;
}

}  // namespace basalt::program::cli
