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
std::string label_of(const std::string& name, bool copy) { return copy ? name + " copy" : name; }
}  // namespace

LoadSummary::Target& LoadSummary::target(const std::string& name, bool copy) {
    for (auto& t : targets_)
        if (t.name == name && t.copy == copy) return t;
    targets_.emplace_back();
    targets_.back().name = name;
    targets_.back().copy = copy;
    return targets_.back();
}

std::string LoadSummary::running() const {
    std::string list;
    unsigned long long d = 0, p = 0;
    for (const auto& t : targets_) {
        if (t.start < 0 || t.reported || t.size() == 0) continue;
        list += (list.empty() ? "" : ", ") + label_of(t.name, t.copy) + " " + gib(t.have()) + " / " + gib(t.size()) + " GiB";
        d += t.have();
        p += t.size();
    }
    return list.empty() ? "" : "loading " + list + " (" + pct(d, p) + ")";
}

std::string LoadSummary::all_targets(double now, bool final) const {
    std::string list;
    unsigned long long d = 0, whole = total_;
    for (const auto& t : targets_) {
        if (t.start < 0 || t.size() == 0) continue;
        list += (list.empty() ? "" : kDot) + label_of(t.name, t.copy) + " " + gib(t.have()) + "/" + gib(t.size()) + " GiB";
        if (!t.copy) {
            d += t.have();
            if (!total_) whole += t.size();
        }
    }
    return list.empty() ? "" : "loading " + list + kDot + (final ? "100%" : pct(d, whole)) + kDot + secs(now - t0_);
}

std::string LoadSummary::redraw(double now) {
    const std::string s = all_targets(now, false);
    if (s.empty()) return "";
    drawn_ = true;
    last_out_ = now;
    return std::string(kClear) + label_ + ": " + s;
}

std::string LoadSummary::feed(const std::string& raw, double now) {
    static const std::string tag = "basalt load: ";
    basalt::log::Parsed p;
    if (basalt::log::parse(raw, p) && p.level == basalt::log::Level::error) {
        const std::string pre = drawn_ ? kClear : "";
        drawn_ = false;
        last_out_ = now;
        return pre + line("engine: " + p.text);
    }
    if (raw.rfind(tag, 0) != 0) return "";
    std::istringstream is(raw.substr(tag.size()));
    std::vector<std::string> w;
    for (std::string x; is >> x;) w.push_back(x);
    if (w.empty() || w[0] == "kv") return "";

    unsigned long long a = 0, b = 0;
    if (w[0] == "total") {
        if (w.size() == 2 && number(w[1], a)) total_ = a;
        return "";
    }
    if (w[0] == "plan") {
        const bool copy = w.size() == 4 && w[1] == "copy";
        if ((w.size() == 3 || copy) && number(w.back(), a)) target(w[copy ? 2 : 1], copy).plan = a;
        return "";
    }
    const bool copy = w[0] == "copy";
    if (copy) w.erase(w.begin());
    if (w.size() != 3 || !number(w[1], a) || !number(w[2], b)) return "";

    Target& t = target(w[0], copy);
    std::string out;
    if (t.start < 0) {
        t.start = now;
        // phases run one after another: a target whose last phase ended is done, whatever its plan said
        for (auto& o : targets_)
            if (&o != &t && o.copy == copy && o.start >= 0 && !o.reported && o.seen && o.last >= o.seen) {
                o.reported = true;
                if (!tty_) out += line("loading " + label_of(o.name, o.copy) + " " + gib(o.have()) + " / " + gib(o.size()) + " GiB done (" + secs(o.end - o.start) + ")");
            }
    }
    t.end = now;
    if (t.seen != b || a < t.last) {   // a new phase: the previous one is complete
        t.done += t.seen ? t.last : 0;
        t.last = 0;
        t.seen = b;
        t.reported = false;
    }
    t.last = a < b ? a : b;
    if (b == 0) return out;

    const bool complete = a >= b && (t.plan == 0 || t.have() >= t.plan);
    if (tty_) {
        if (!drawn_ || now - last_out_ >= redraw_ || complete) return redraw(now);
        return "";
    }
    if (!header_) {
        header_ = true;
        std::string parts;
        for (const auto& x : targets_)
            if (x.plan > 0) parts += (parts.empty() ? "" : ", ") + label_of(x.name, x.copy) + " " + gib(x.have()) + " / " + gib(x.plan) + " GiB";
        out += line("loading" + (total_ ? " " + gib(total_) + " GiB" : std::string()) +
                    (parts.empty() ? "" : (total_ ? ": " : " ") + parts));
        last_out_ = now;
    }
    if (complete && !t.reported) {
        t.reported = true;
        out += line("loading " + label_of(t.name, t.copy) + " " + gib(t.have()) + " / " + gib(t.size()) + " GiB done (" +
                    secs(now - t.start) + ")");
        last_out_ = now;
    }
    return out;
}

std::string LoadSummary::tick(double now) {
    if (tty_) return drawn_ && now - last_out_ >= redraw_ ? redraw(now) : "";
    if (now - last_out_ < period_) return "";
    const std::string s = running();
    if (s.empty()) return "";
    last_out_ = now;
    return line(s);
}

std::string LoadSummary::finish(double now, bool ready) {
    std::string out;
    if (tty_ && drawn_) {
        for (auto& t : targets_) t.reported = t.reported || ready;
        out = std::string(kClear) + label_ + ": " + all_targets(now, ready) + "\n";
        drawn_ = false;
    }
    if (ready) out += line("engine ready in " + secs(now - t0_));
    return out;
}

}  // namespace basalt::program
