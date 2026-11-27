// tests/program/prefix_anchor_test.cpp - a held conversation met against a prompt: the model's own pieces stand for
// the bytes the prompt spells (header-only, CPU).
#include "program/serve/prefix_anchor.hpp"

#include <cstdio>
#include <map>
#include <string>
#include <vector>

using basalt::program::serve::HeldMeet;
using basalt::program::serve::meet_held;
using basalt::program::serve::respell_before_media;
using basalt::program::serve::respell_fits;
using basalt::program::serve::shared_start;
using basalt::program::serve::slot_cover;
using basalt::program::serve::splice_held;
using Ids = std::vector<int32_t>;

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}
// ids 1-9 spell bytes; 3 is "ab", what the tokenizer makes of a+b; 10 and 11 open a prompt; 12 is a picture's id, no bytes
const std::map<int32_t, std::string> kBytes = {{1, "a"}, {2, "b"}, {3, "ab"}, {4, "<end>"}, {5, "\n"}, {6, "c"}, {7, "\n\n"},
                                               {10, "<sys>"}, {11, "hi"}};
std::string_view bytes(int32_t id) {
    const auto it = kBytes.find(id);
    return it == kBytes.end() ? std::string_view() : std::string_view(it->second);
}
HeldMeet meet(const Ids& prompt, const Ids& held, bool respell = true) {
    return meet_held(prompt.data(), (int64_t) prompt.size(), held.data(), (int64_t) held.size(), respell, bytes);
}
int64_t cover(const Ids& prompt, const Ids& held, bool respell = true) {
    return slot_cover(prompt.data(), (int64_t) prompt.size(), held.data(), (int64_t) held.size(),
                      shared_start(prompt.data(), (int64_t) prompt.size(), held.data(), (int64_t) held.size()),
                      respell, bytes);
}
}  // namespace

int main() {
    {   // the model wrote a, b; the prompt reads ab: the slot's ids stand for the prompt's
        const Ids held = {10, 11, 1, 2}, prompt = {10, 11, 3, 4, 5, 6};
        const HeldMeet m = meet(prompt, held);
        check(m.held == 4 && m.resume == 3 && m.respelled(), "the model's pieces stand for the prompt's token with the same bytes");
        Ids out;
        splice_held(held.data(), m, prompt.data(), (int64_t) prompt.size(), out);
        check(out == Ids({10, 11, 1, 2, 4, 5, 6}), "the prompt becomes the held ids, then its own tokens from the boundary");
    }
    {
        // The reply's text is respelled, then an image is added later in the prompt.
        const Ids held = {10, 11, 1, 2}, prompt = {10, 11, 3, 4, 5, 12, 6};
        const HeldMeet m = meet(prompt, held);
        check(m.held == 4 && m.resume == 3 && m.respelled(), "a held text reply meets before the later image");
        check(respell_before_media(m, 5) && respell_before_media(m, 3) && respell_before_media(m, -1),
              "an image at or after the resume boundary does not block respelling");
        check(!respell_before_media(m, 2), "an image before the resume boundary blocks respelling");
        Ids out;
        splice_held(held.data(), m, prompt.data(), (int64_t) prompt.size(), out);
        check(out == Ids({10, 11, 1, 2, 4, 5, 12, 6}), "respelling preserves the later image and the text after it");
    }
    {   // two tokens of the model's one newline pair against the tokenizer's one token
        const Ids held = {10, 5, 5}, prompt = {10, 7, 6};
        const HeldMeet m = meet(prompt, held);
        check(m.held == 3 && m.resume == 2, "two newlines the model wrote stand for the tokenizer's one token");
    }
    {   // a split in one place and a merge in another: as many tokens, other ids
        const Ids held = {10, 1, 2, 6, 7}, prompt = {10, 3, 6, 5, 5, 4};
        const HeldMeet m = meet(prompt, held);
        check(m.held == 5 && m.resume == 5 && m.respelled(), "as many tokens but other ids is still a respelling");
        Ids out;
        splice_held(held.data(), m, prompt.data(), (int64_t) prompt.size(), out);
        check(out == Ids({10, 1, 2, 6, 7, 4}), "the held ids replace the prompt's tokens of the same bytes");
    }
    check(meet({10, 11, 3, 4}, {10, 11, 1, 6}).held == 0, "other bytes: the prompt is as it is");
    check(meet({10, 11, 3, 4}, {10, 11, 1}).held == 0, "a prompt token across the held ids' end: the prompt is as it is");
    {
        const HeldMeet m = meet({10, 11, 3, 4}, {10, 11, 3});
        check(m.held == 3 && m.resume == 3 && !m.respelled(), "a held conversation that is the prompt's start changes nothing");
    }
    check(meet({10, 11, 3}, {10, 11, 3}).held == 0, "a held conversation is never the whole prompt");
    check(meet({10, 11, 3}, {10, 11, 3, 4}).held == 0, "nor longer than it");
    check(meet({10, 7, 6}, {10, 5, 5}).held == 3, "held ids that spell fewer tokens than they are meet a prompt that goes on past them");
    check(meet({10, 11, 3}, {10, 11, 1, 2}).held == 0, "the held ids must end before the prompt's last token");
    check(meet({3, 4, 5}, {1, 2}).held == 0, "nothing shared before the reply: another conversation");
    check(meet({10, 11, 3, 4}, {10, 11, 1, 2}, false).held == 0, "without a tokenizer table only a prefix meets");
    check(meet({10, 11, 3, 4}, {10, 11, 1, 12}).held == 0, "an id without bytes (a picture's): the prompt is as it is");
    check(meet({10, 11, 12, 4}, {10, 11, 1, 2}).held == 0, "a prompt id without bytes: the prompt is as it is");
    {   // a reply of 1<<17 + 1 pieces is another conversation, not a reply sent back
        Ids held = {10, 11}, prompt = {10, 11};
        for (int i = 0; i < (1 << 17) + 1; ++i) held.push_back(5);
        for (int i = 0; i < (1 << 17) + 1; ++i) prompt.push_back(5);
        prompt[2] = 3;
        prompt.push_back(4);
        check(meet(prompt, held).held == 0, "more than a reply's worth past the shared start: as it is");
    }
    {   // the slot that holds the conversation is told from one that does not
        const Ids prompt = {10, 11, 3, 4, 5, 6};
        const Ids slot_a = {10, 11, 1, 2}, slot_b = {10, 99, 1, 2};
        check(meet(prompt, slot_a).held == 4 && meet(prompt, slot_b).held == 0, "the slot with the conversation meets the prompt, the other does not");
    }
    {   // the slot a conversation sits in: the ids its held conversation continues, not the head two conversations share
        check(cover({10, 11, 1, 6, 5, 4}, {10, 11, 1, 2}) == 0,
              "a shared start whose bytes part: another conversation, not the slot's");
        check(cover({10, 11, 6, 5}, {10, 11, 3, 4, 5}) == 0, "a system prompt two conversations share is no conversation of the slot's");
        check(cover({10, 11, 3}, {10, 11, 1, 2}) == 2,
              "the reply the client sent back, the tokenizer's last token spilling past the held ids: still the slot's");
        check(cover({10, 11, 3, 4, 5, 6}, {10, 11, 1, 2}) == 4, "a respelled reply covers all its held ids");
        check(cover({10, 11, 3}, {10, 11, 3}) == 2, "a slot holding the whole prompt covers it but for its last token");
        check(cover({10, 11, 3}, {10, 11, 3, 4}) == 2, "nor does one holding more of the conversation");
        check(cover({3, 4, 5}, {1, 2}) == 0, "another conversation covers nothing");
        check(cover({10, 11, 1, 6, 5, 4}, {10, 11, 1, 2}, false) == 0, "without a tokenizer table only a whole prefix continues");
    }
    check(respell_fits(7, 6, 16, 8 + 7 + 16), "a longer prompt that leaves max_new room is taken");
    check(!respell_fits(7, 6, 16, 8 + 6 + 16), "a longer prompt that leaves no max_new room is not");
    check(respell_fits(6, 6, 16, 8 + 6 + 16 - 100), "a prompt that does not grow always fits");
    return failures == 0 ? 0 : 1;
}
