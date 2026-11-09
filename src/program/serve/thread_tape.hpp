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
