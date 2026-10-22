// include/basalt/program/engine_session.hpp - the engine as a child process driven over its serve protocol (GEN
// requests, T / DONE lines, one-line commands), plus the argument and fixture helpers the benchmark programs share.
#pragma once

#include "basalt/platform/process.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace basalt::program {

/// One finished GEN request, from its DONE line.
struct Done {
    bool ok = false;
    std::string err;
    long long generated = 0, prompt = 0, accepted = 0, offered = 0, reused = 0, read_n = -1;
    double prompt_ms = 0, decode_ms = 0, wall_s = 0, first_token_s = -1;
    std::string finish;
};

/// The engine started with `--serve`; stdin / stdout are the protocol, stderr goes to a log file.
class Engine {
public:
    ~Engine() { stop(); }

    /// Starts the engine and waits for READY, printing a summary of its load progress meanwhile; `info` holds the key=value pairs of its INFO line.
    bool start(const std::string& engine_bin, const std::vector<std::string>& args, const std::string& log_path,
               std::string& err);

    /// One GEN request. `keys` are the sampling keys, each with a leading space.
    bool request(const std::string& keys, long long max_new, const std::vector<int64_t>& ids, Done& d);

    /// Sends one command line and collects the reply lines up to and including the first that starts with
    /// `end_prefix`. False (with `err`) on an ERR line or when the engine exits first.
    bool command(const std::string& line, const std::string& end_prefix, std::vector<std::string>& reply,
                 std::string& err);

    /// QUIT and wait: the engine flushes its logs on a normal exit.
    void stop();

    std::map<std::string, std::string> info;
    /// Prefix of the load-progress lines `start` prints to stderr while it waits for READY.
    std::string label = "basalt-bench";

private:
    bool write_all(const std::string& s);
    bool read_line(std::string& line);
    void reap();
    void finish_reap();
    basalt::platform::PipedChild child_;
    std::string buf_, log_path_;
    size_t rpos_ = 0;
};

std::vector<std::string> split(const std::string& s, char sep);
bool all_digits(const std::string& s);
/// "16384", "32k" to tokens: k is 1024, except a number that rounds in thousands to a power of two is that power
/// ("262k" is 262144).
bool parse_depth(const std::string& s, long long& tokens);

/// FLAG VALUE: replace the value when the flag is there, else append both.
void set_arg(std::vector<std::string>& args, const std::string& flag, const std::string& value);
bool has_arg(const std::vector<std::string>& args, const std::string& flag);
std::string arg_value(const std::vector<std::string>& args, const std::string& flag);
/// FLAG [VALUE] words added to `args`: a flag already there takes the new value, else both are appended.
void merge_args(std::vector<std::string>& args, const std::vector<std::string>& add);

/// The engine's own `basalt error:` lines from its log, else the last few lines.
std::string engine_failure(const std::string& log_path);

/// A fixture file: one comma-separated token-id list per line.
bool read_id_lines(const std::string& path, std::vector<std::vector<int64_t>>& lines, std::string& err);

}  // namespace basalt::program
