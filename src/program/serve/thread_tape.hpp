// Native thread token tapes. Host-only; no CUDA headers, no implicit KV trust.
// The ledger is authoritative: tapes are disposable and must be reattached after restart.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace basalt::program::serve {

class NativeThreadTapes {
public:
    struct Result {
        bool ok;
        const char* why;
    };
    explicit NativeThreadTapes(size_t max_threads = 64) : max_threads_(max_threads) {}

    static bool valid_name(const std::string& s) {
        if (s.empty() || s.size() > 128) return false;
        return std::all_of(s.begin(), s.end(), [](unsigned char c) {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                   (c >= '0' && c <= '9') || c == '-' || c == '_';
        });
    }
    Result attach(const std::string& id, const std::string& head,
                  const std::vector<int32_t>& tokens, size_t context_limit, int64_t vocab) {
        if (!valid_name(id) || !valid_name(head) || !valid_tokens(tokens,vocab) ||
            tokens.empty() || context_limit == 0 || tokens.size() >= context_limit)
            return {false, "invalid attach identity or token prefix"};
        auto it = tapes_.find(id);
        if (it != tapes_.end()) {
            return it->second.head == head && it->second.ids == tokens
                ? Result{true,"already attached"} : Result{false,"attach conflict; release before replacing"};
        }
        if (tapes_.size() >= max_threads_) return {false,"thread tape capacity reached"};
        tapes_.emplace(id, Entry{head,tokens});
        return {true,"attached"};
    }
    Result append(const std::string& id, const std::string& expected,
                  const std::string& next, const std::vector<int32_t>& suffix,
                  size_t context_limit, int64_t vocab) {
        auto it=tapes_.find(id);
        if (it == tapes_.end()) return {false,"unknown thread tape"};
        Entry& e=it->second;
        if (!valid_name(expected) || !valid_name(next) || suffix.empty() ||
            !valid_tokens(suffix,vocab)) return {false,"invalid append"};
        if (e.head != expected) return {false,"stale thread tape head"};
        if (next == expected) return {false,"new head must differ"};
        if (context_limit == 0 || suffix.size() >= context_limit ||
            e.ids.size() >= context_limit - suffix.size())
            return {false,"thread tape context overflow"};
        e.ids.insert(e.ids.end(),suffix.begin(),suffix.end());
        e.head=next;
        return {true,"appended"};
    }
    // Fork: the child starts as a copy of the parent's tape at `keep` tokens (0 = whole tape).
    // The child shares the token prefix, so the exact-prefix KV check reuses the parent's cache.
    // Same child id + same result is idempotent; a different result is a conflict.
    Result fork(const std::string& parent, const std::string& parent_head,
                const std::string& child, const std::string& child_head, size_t keep) {
        if (!valid_name(parent) || !valid_name(child) || !valid_name(parent_head) ||
            !valid_name(child_head) || parent == child)
            return {false, "invalid fork identity"};
        auto p = tapes_.find(parent);
        if (p == tapes_.end()) return {false, "unknown thread tape"};
        if (p->second.head != parent_head) return {false, "stale thread tape head"};
        if (keep == 0) keep = p->second.ids.size();
        if (keep > p->second.ids.size()) return {false, "fork point beyond tape"};
        std::vector<int32_t> ids(p->second.ids.begin(), p->second.ids.begin() + (std::ptrdiff_t) keep);
        auto c = tapes_.find(child);
        if (c != tapes_.end())
            return c->second.head == child_head && c->second.ids == ids
                ? Result{true, "already forked"} : Result{false, "fork conflict; release before replacing"};
        if (tapes_.size() >= max_threads_) return {false, "thread tape capacity reached"};
        tapes_.emplace(child, Entry{child_head, std::move(ids)});
        return {true, "forked"};
    }
    // Length of the common token prefix of two tapes (what a fork pair can share in KV).
    size_t shared_prefix(const std::string& a, const std::string& b) const {
        auto x = tapes_.find(a), y = tapes_.find(b);
        if (x == tapes_.end() || y == tapes_.end()) return 0;
        const auto& u = x->second.ids; const auto& v = y->second.ids;
        size_t n = 0;
        while (n < u.size() && n < v.size() && u[n] == v[n]) ++n;
        return n;
    }
    const std::vector<int32_t>* get(const std::string& id, const std::string& head) const {
        auto it=tapes_.find(id);
        if (it==tapes_.end() || it->second.head!=head) return nullptr;
        return &it->second.ids;
    }
    bool release(const std::string& id) { return tapes_.erase(id)>0; }
    size_t size() const {return tapes_.size();}
private:
    struct Entry {
        std::string head;
        std::vector<int32_t> ids;
    };
    static bool valid_tokens(const std::vector<int32_t>& tokens,int64_t vocab) {
        if (vocab<=0) return false;
        return std::all_of(tokens.begin(),tokens.end(),[vocab](int32_t t){
            return t>=0 && static_cast<int64_t>(t)<vocab;
        });
    }
    std::map<std::string,Entry> tapes_;
    size_t max_threads_;
};
} // namespace basalt::program::serve
