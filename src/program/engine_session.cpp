// src/program/engine_session.cpp - see engine_session.hpp.
#include "basalt/log.hpp"
#include "basalt/program/engine_session.hpp"

#include "basalt/platform/file.hpp"
#include "basalt/platform/system.hpp"
#include "basalt/program/load_summary.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>
#include <cmath>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace basalt::program {

namespace {
using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

// Follows the engine's log file while the start waits for READY and prints the load summary.
class LogFollower {
public:
    LogFollower(const std::string& path, const std::string& label)
        : path_(path), t0_(Clock::now()), sum_(label, basalt::platform::stderr_is_terminal(), 0.0),
          thread_([this] { run(); }) {}
    ~LogFollower() { stop(false); }
    void stop(bool ready) {
        if (stopped_) return;
        stopped_ = true;
        {
            std::lock_guard<std::mutex> lk(mu_);
            quit_ = true;
        }
        cv_.notify_all();
        thread_.join();
        poll();
        emit(sum_.finish(now(), ready));
    }

private:
    double now() const { return secs(t0_, Clock::now()); }
    static void emit(const std::string& s) {
        if (s.empty()) return;
        basalt::log::raw_text(s);
    }
    void poll() {
        std::ifstream f(path_, std::ios::binary);
        if (f) {
            f.seekg(offset_);
            char tmp[16384];
            while (f.read(tmp, sizeof tmp) || f.gcount() > 0) {
                partial_.append(tmp, (size_t) f.gcount());
                offset_ += (std::streamoff) f.gcount();
            }
        }
        size_t nl;
        while ((nl = partial_.find('\n')) != std::string::npos) {
            std::string l = partial_.substr(0, nl);
            partial_.erase(0, nl + 1);
            if (!l.empty() && l.back() == '\r') l.pop_back();
            emit(sum_.feed(l, now()));
        }
        emit(sum_.tick(now()));
    }
    void run() {
        std::unique_lock<std::mutex> lk(mu_);
        while (!quit_) {
            lk.unlock();
            poll();
            lk.lock();
            cv_.wait_for(lk, std::chrono::milliseconds(200), [this] { return quit_; });
        }
    }

    std::string path_, partial_;
    Clock::time_point t0_;
    basalt::program::LoadSummary sum_;
    std::streamoff offset_ = 0;
    std::mutex mu_;
    std::condition_variable cv_;
    bool quit_ = false, stopped_ = false;
    std::thread thread_;
};
}  // namespace

bool Engine::start(const std::string& engine_bin, const std::vector<std::string>& args, const std::string& log_path,
                   std::string& err) {
    std::vector<std::string> argv = {engine_bin, "--serve"};
    argv.insert(argv.end(), args.begin(), args.end());
    if (!basalt::platform::spawn_piped(argv, log_path, child_, err)) return false;
    basalt::platform::set_stop_child(child_.child);
    log_path_ = log_path;
    std::string line;
    LogFollower follower(log_path, label);
    while (read_line(line)) {
        if (line.rfind("INFO ", 0) == 0) {
            std::istringstream is(line.substr(5));
            std::string kv;
            while (is >> kv) {
                const size_t eq = kv.find('=');
                if (eq != std::string::npos) info[kv.substr(0, eq)] = kv.substr(eq + 1);
            }
        } else if (line.rfind("READY", 0) == 0) {
            follower.stop(true);
            return true;
        }
    }
    follower.stop(false);
    err = "the engine exited before READY" + engine_failure(log_path);
    reap();
    return false;
}

bool Engine::request(const std::string& keys, long long max_new, const std::vector<int64_t>& ids, Done& d) {
    d = Done{};
    std::string msg = "GEN " + std::to_string(max_new) + keys + " ";
    msg.reserve(msg.size() + ids.size() * 7 + 2);
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) msg += ',';
        msg += std::to_string(ids[i]);
    }
    msg += '\n';
    const auto t0 = Clock::now();
    if (!write_all(msg)) { d.err = "the engine closed its input"; return false; }
    std::string line;
    while (read_line(line)) {
        if (line.rfind("T ", 0) == 0) {
            if (d.first_token_s < 0) d.first_token_s = secs(t0, Clock::now());
        } else if (line.rfind("DONE", 0) == 0) {
            d.wall_s = secs(t0, Clock::now());
            std::istringstream is(line);
            std::string tag;
            std::vector<std::string> f;
            while (is >> tag) f.push_back(tag);
            if (f.size() < 8) { d.err = "short DONE line: " + line; return false; }
            d.generated = std::atoll(f[1].c_str());
            d.prompt = std::atoll(f[2].c_str());
            d.prompt_ms = std::atof(f[3].c_str());
            d.decode_ms = std::atof(f[4].c_str());
            d.finish = f[5];
            d.accepted = std::atoll(f[6].c_str());
            d.offered = std::atoll(f[7].c_str());
            if (f.size() > 8) d.reused = std::atoll(f[8].c_str());
            if (f.size() > 14) d.read_n = std::atoll(f[14].c_str());
            d.ok = true;
            return true;
        } else if (line.rfind("ERR", 0) == 0) {
            d.err = line;
            return false;
        }
    }
    d.err = "the engine exited during the request (see " + log_path_ + ")";
    return false;
}

bool Engine::command(const std::string& line, const std::string& end_prefix, std::vector<std::string>& reply,
                     std::string& err) {
    reply.clear();
    if (!write_all(line + "\n")) { err = "the engine closed its input"; return false; }
    std::string l;
    while (read_line(l)) {
        if (l.rfind("ERR", 0) == 0) { err = l; return false; }
        reply.push_back(l);
        if (l.rfind(end_prefix, 0) == 0) return true;
    }
    err = "the engine exited during " + line + " (see " + log_path_ + ")";
    return false;
}

void Engine::stop() {
    if (child_.child.id <= 0 && child_.child.handle == nullptr) return;
    const std::string q = "QUIT\n";
    if (child_.in_fd >= 0) {
        (void) basalt::platform::pipe_write_all(child_.in_fd, q.data(), q.size());
        basalt::platform::close_fd(child_.in_fd);
        child_.in_fd = -1;
    }
    for (int i = 0; i < 600; ++i) {
        if (basalt::platform::child_wait(child_.child, 100) >= 0) { finish_reap(); return; }
    }
    basalt::platform::child_terminate(child_.child);
    reap();
}

bool Engine::write_all(const std::string& s) { return basalt::platform::pipe_write_all(child_.in_fd, s.data(), s.size()); }

bool Engine::read_line(std::string& line) {
    for (;;) {
        const size_t nl = buf_.find('\n', rpos_);
        if (nl != std::string::npos) {
            line.assign(buf_, rpos_, nl - rpos_);
            rpos_ = nl + 1;
            if (rpos_ > (1u << 20)) { buf_.erase(0, rpos_); rpos_ = 0; }
            return true;
        }
        char tmp[65536];
        const int64_t n = basalt::platform::pipe_read(child_.out_fd, tmp, sizeof tmp);
        if (n <= 0) return false;
        buf_.append(tmp, (size_t) n);
    }
}

void Engine::reap() {
    // wait for it, up to the time a normal exit needs, then let go whether or not it ended
    (void) basalt::platform::child_wait(child_.child, 10000);
    finish_reap();
}

void Engine::finish_reap() {
    if (child_.in_fd >= 0) basalt::platform::close_fd(child_.in_fd);
    if (child_.out_fd >= 0) basalt::platform::close_fd(child_.out_fd);
    child_.in_fd = child_.out_fd = -1;
    basalt::platform::clear_stop_child();
    basalt::platform::child_close(child_.child);
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> r;
    std::string item;
    std::istringstream is(s);
    while (std::getline(is, item, sep))
        if (!item.empty()) r.push_back(item);
    return r;
}

bool all_digits(const std::string& s) {
    return !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
}

bool parse_depth(const std::string& s, long long& tokens) {
    std::string d = s;
    std::transform(d.begin(), d.end(), d.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    if (d.empty()) return false;
    char* e = nullptr;
    if (d.back() == 'k') {
        const double v = std::strtod(d.c_str(), &e);
        if (e != d.c_str() + d.size() - 1 || v <= 0) return false;
        tokens = (long long) (v * 1024);
        // "262k" names the 262144 context: a number that rounds, in thousands, to a power of two is that power
        for (long long p = 32768; p <= (1ll << 30); p *= 2)
            if (std::llround((double) p / 1000.0) == std::llround(v)) tokens = p;
    } else {
        if (!all_digits(d)) return false;
        tokens = std::atoll(d.c_str());
    }
    return tokens > 0;
}

// FLAG VALUE: replace the value when the flag is there, else append both.
void set_arg(std::vector<std::string>& args, const std::string& flag, const std::string& value) {
    for (size_t i = 0; i + 1 < args.size(); ++i)
        if (args[i] == flag) { args[i + 1] = value; return; }
    args.push_back(flag);
    args.push_back(value);
}

bool has_arg(const std::vector<std::string>& args, const std::string& flag) {
    return std::find(args.begin(), args.end(), flag) != args.end();
}

std::string arg_value(const std::vector<std::string>& args, const std::string& flag) {
    for (size_t i = 0; i + 1 < args.size(); ++i)
        if (args[i] == flag) return args[i + 1];
    return "";
}

// FLAG [VALUE] words added to `args`: a flag already there takes the new value, else both are appended.  A word
// that does not begin with "--" after a flag is its value.
void merge_args(std::vector<std::string>& args, const std::vector<std::string>& add) {
    for (size_t i = 0; i < add.size(); ++i) {
        const bool value = i + 1 < add.size() && add[i + 1].rfind("--", 0) != 0;
        if (add[i].rfind("--", 0) != 0) { args.push_back(add[i]); continue; }
        if (value) { set_arg(args, add[i], add[i + 1]); ++i; }
        else if (!has_arg(args, add[i])) args.push_back(add[i]);
    }
}

// The engine's own error lines from its log, else the last few lines.
std::string engine_failure(const std::string& log_path) {
    std::ifstream f(log_path);
    std::vector<std::string> lines, errors;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        lines.push_back(line);
        basalt::log::Parsed p;
        if (basalt::log::parse(line, p) && p.level == basalt::log::Level::error) errors.push_back(line);
    }
    if (errors.empty() && lines.size() > 8) lines.erase(lines.begin(), lines.end() - 8);
    std::string r;
    for (const auto& l : errors.empty() ? lines : errors) r += "\n  " + l;
    return r;
}

bool read_id_lines(const std::string& path, std::vector<std::vector<int64_t>>& lines, std::string& err) {
    std::ifstream f(path);
    if (!f) { err = "cannot read " + path; return false; }
    std::string line;
    while (std::getline(f, line)) {
        std::vector<int64_t> v;
        const char* p = line.c_str();
        char* e = nullptr;
        for (;;) {
            while (*p == ',' || *p == ' ' || *p == '\t' || *p == '\r') ++p;
            if (*p == '\0') break;
            const long long x = std::strtoll(p, &e, 10);
            if (e == p) { err = path + ": not a list of token ids"; return false; }
            v.push_back(x);
            p = e;
        }
        if (!v.empty()) lines.push_back(std::move(v));
    }
    return true;
}

}  // namespace basalt::program
