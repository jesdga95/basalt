// include/basalt/platform/process.hpp - starting a child and stopping one, so Win32 stays out of the engine.
//
// The engine wants two shapes. `exec_replacing` is what `basalt serve` does: the child takes this process's place,
// and this one is gone when the child ends. `spawn_and_wait` is what a tool that runs a child to completion and
// keeps its own process wants. A caller that starts the child itself - because it is wiring the child's standard
// descriptors to pipes it owns - gets `Child` plus the wait/stop calls and the signal helpers, which is the Ctrl+C
// shape a bench run already uses: one child of record, terminated with the run.
//
// There is no process-stats reading here. Windows has none to move: the engine's stall report reads /proc and says
// nothing there today, so a wrapper would be new behaviour rather than the same behaviour spelled portably.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::platform {

/// Quotes one argument for a platform that takes one command line rather than a vector: backslashes before a quote
/// are doubled, a run of backslashes before the closing quote is doubled, and anything with a space or a quote is
/// wrapped. Linux hands arguments as a vector and needs no quoting, but the function is built there too so a call
/// site can keep one spelling.
std::string quote_arg(const std::string& arg);

/// A child process this process started, as far as waiting for it and stopping it goes. The caller fills it in from
/// whatever it started the child with; `handle` is the process handle where the OS has one and is unused elsewhere.
struct Child {
    int64_t id = -1;          ///< the process id, on both platforms
    void* handle = nullptr;   ///< where the OS names the child by a handle
};

/// Waits for the child to end. `timeout_ms` of 0 only asks whether it already has, -1 waits as long as it takes.
/// Returns the child's exit code, or -1 when it is still running at the timeout. A child stopped by a signal
/// reports 128 + the signal number, which is how a shell spells it. Linux has reaped the child when this returns a
/// code; Windows leaves the handle until `child_close`.
int child_wait(Child& c, int timeout_ms);

/// Stops a child that will not end by itself. It is still a child until it is waited on.
void child_terminate(Child& c);

/// Lets go of what the Child holds. Safe to call twice.
void child_close(Child& c);

/// Names the child a stop signal should end (the run's engine), or says there is none.
void set_stop_child(const Child& c);
void clear_stop_child();

/// Installs the handler that ends that child and leaves with 130, for SIGINT and SIGTERM. A handler installed here
/// replaces the caller's own, so install it once, at the start of a run.
void install_child_stop_signals();

/// Makes a closed pipe surface as a failed write rather than ending the process. There is no SIGPIPE to ignore on
/// Windows, so this says nothing there.
void ignore_broken_pipe();

/// Sets an environment variable of this process, which a child started from it inherits. The two platforms spell the
/// call differently; neither keeps a value that is already there, so a caller that means "only when unset" asks
/// `getenv` first.
void set_env(const std::string& name, const std::string& value);

/// Runs `argv` (argv[0] is the program, found on PATH as execvp finds it) to completion in a child process and
/// returns the child's exit code. -1 when the child could not be started, with errno set: ENOENT for a program that
/// is not there, EINVAL for an empty argv.
int spawn_and_wait(const std::vector<std::string>& argv);

/// Gives `argv` this process's place. It does not return when it works. -1 with errno set when the program cannot
/// be run, which is the caller's message. Linux replaces the process; Windows has no exec, so the child is run,
/// waited on, and this process ends with the child's exit code - which is what makes a wrapper's exit status mean
/// what the child's did.
int exec_replacing(const std::vector<std::string>& argv);

}  // namespace basalt::platform
