// include/basalt/platform/system.hpp - what the engine asks the host system, so the engine itself never names one.
//
// Each call is the same question on every platform; where a platform has nothing to answer with, the call says so in
// its return value (an empty path, -1, a zero count) rather than the caller testing which OS it is on.
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace basalt::platform {

/// This process's id.
int64_t process_id();

/// The kind of system the program runs on, for a benchmark row: "Linux", "Windows", or "WSL2" for Linux inside the
/// Windows subsystem.
std::string host_os_name();

/// The Python the server runs under: the virtual environment in `src_dir` when it has one, else the interpreter name
/// the platform puts on PATH.
std::string python_interpreter(const std::string& src_dir);

/// Asks that this process be the one ended first if the machine runs out of memory. No effect where the OS has no
/// such knob.
void prefer_oom_victim();

/// Whether a process is given direct GPU-to-GPU access between its devices, which a second GPU holding an expert tier
/// needs. False where the platform withholds it.
bool gpu_peer_access();

/// Puts the three standard streams in binary mode where the C runtime defaults them to text, which would turn each
/// "\n" the serve protocol writes into "\r\n" and fold a "\r\n" in a request to "\n". For a process whose standard
/// streams are pipes carrying bytes; a console wants its text mode. Nothing to do where text and binary are the same.
void set_stdio_binary();

/// Whether standard error is a terminal that takes ANSI line control (a status line rewritten in place); false for a
/// pipe or a file.
bool stderr_is_terminal();

/// Where the OS keeps its memory accounting as text files, for the readers that parse them: the system's
/// `meminfo`, this process's cgroup membership and the cgroup mount. Empty where there is no such file, and the OS
/// call (`available_memory`) is the reading.
std::string meminfo_path();
std::string self_cgroup_path();
std::string cgroup_root();

/// This process's own memory use, for the stall report. Every field is 0 where the OS gives no reading.
struct ProcessMemory {
    unsigned long long faults = 0;       ///< hard page faults
    unsigned long long rss_mib = 0;      ///< resident set
    unsigned long long avail_mib = 0;    ///< RAM the system can hand out
    unsigned long long commit_mib = 0;   ///< what is swapped out
};
ProcessMemory process_memory();

/// Whether the file at `path` is on a rotational disk (1), not (0) or unknown (-1); `dev` names the disk when known.
int disk_rotational(const std::string& path, std::string& dev);

/// The threads of this process in uninterruptible sleep (almost always waiting for the disk), counted by where in the
/// kernel they wait. `threads` is 0 where the OS has no such listing.
struct ThreadWaits {
    int waiting = 0, threads = 0;
    std::map<std::string, int> where;
};
ThreadWaits threads_in_disk_wait();

/// What a program writes to its standard output, run with `argv` (argv[0] found on PATH) and read to the end. With
/// `include_stderr` its standard error is mixed in, otherwise it is thrown away. nullopt when the program cannot be
/// started.
std::optional<std::string> capture_output(const std::vector<std::string>& argv, bool include_stderr);

}  // namespace basalt::platform
