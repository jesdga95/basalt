// src/program/load_summary.cpp - see load_summary.hpp.
#include "basalt/program/load_summary.hpp"

#include "basalt/log.hpp"

#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace basalt::program {

namespace {
const char* const kClear = "\r\x1b[2K";
const char* const kDot = " \xC2\xB7 ";

bool number(const std::string& s, unsigned long long& v) {
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos) return false;
    v = std::strtoull(s.c_str(), nullptr, 10);
    return true;
}
std::string secs(double s) { return std::to_string((long long) (s + 0.5)) + " s"; }
std::string pct(unsigned long long a, unsigned long long b) {
    return std::to_string(b ? (int) (a >= b ? 100 : a * 100 / b) : 0) + "%";
}
std::string gib(unsigned long long bytes) {
    char b[32];
    std::snprintf(b, sizeof b, "%.1f", (double) bytes / 1073741824.0);
    return b;
}
}  // namespace

std::string LoadSummary::status(double now, bool final) const {
    const unsigned long long whole = size();
    if (start_ < 0 || whole == 0) return "";
    return "loading weights " + gib(done_) + "/" + gib(whole) + " GiB" + kDot + (final ? "100%" : pct(done_, whole)) + kDot +
           secs(now - t0_);
}

std::string LoadSummary::redraw(double now) {
    const std::string s = status(now, false);
    if (s.empty()) return "";
    drawn_ = true;
    last_out_ = now;
    return std::string(kClear) + label_ + ": " + s;
}

std::string LoadSummary::feed(const std::string& raw, double now) {
    static const std::string tag = "basalt load: ";
    basalt::log::Parsed p;
    if (basalt::log::parse(raw, p) && p.level != basalt::log::Level::info) {   // shown at once; a warning keeps its marker
        const std::string pre = drawn_ ? kClear : "";
        drawn_ = false;
        last_out_ = now;
        return pre + line(std::string(p.level == basalt::log::Level::warning ? "engine: warning: " : "engine: ") + p.text);
    }
    if (raw.rfind(tag, 0) != 0) return "";
    std::istringstream is(raw.substr(tag.size()));
    std::vector<std::string> w;
    for (std::string x; is >> x;) w.push_back(x);
    if (w.empty()) return "";

    unsigned long long a = 0, b = 0;
    if (w[0] == "total") {
        if (w.size() == 2 && number(w[1], a)) total_ = a;
        return "";
    }
    // the weights line is the load; the n-gram table lock and the session state are outside it
    if (w[0] != "weights" || w.size() != 3 || !number(w[1], a) || !number(w[2], b)) return "";

    std::string out;
    plan_ = b;
    done_ = a < b ? a : b;
    if (start_ < 0) start_ = now;
    if (b == 0) return out;
    const bool complete = done_ >= size();
    if (tty_) {
        if (!drawn_ || now - last_out_ >= redraw_ || complete) return redraw(now);
        return "";
    }
    if (!header_) {
        header_ = true;
        const unsigned long long whole = total_ ? total_ : size();
        out += line("loading " + gib(whole) + " GiB");
        last_out_ = now;
    }
    if (complete && !reported_) {
        reported_ = true;
        out += line("loading weights " + gib(done_) + " / " + gib(size()) + " GiB done (" + secs(now - start_) + ")");
        last_out_ = now;
    }
    return out;
}

std::string LoadSummary::tick(double now) {
    if (tty_) return drawn_ && now - last_out_ >= redraw_ ? redraw(now) : "";
    if (now - last_out_ < period_) return "";
    if (start_ < 0 || reported_ || size() == 0) return "";
    last_out_ = now;
    return line("loading " + gib(done_) + " / " + gib(size()) + " GiB (" + pct(done_, size()) + ")");
}

std::string LoadSummary::finish(double now, bool ready) {
    std::string out;
    if (tty_ && drawn_) {
        out = std::string(kClear) + label_ + ": " + status(now, ready) + "\n";
        drawn_ = false;
    }
    if (ready) out += line("engine ready in " + secs(now - t0_));
    return out;
}

}  // namespace basalt::program
