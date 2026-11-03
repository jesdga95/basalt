// basalt/program/grammar/reply_form.hpp - where a model's reply holds its thinking, for the grammar guard.
#pragma once

#include <string>

namespace basalt::program::grammar {

/// The tokens of a reply's thinking, by text. The default is a <think> ... </think> block before the answer (added
/// tokens). A reply made of blocks names `header` too: every block opens with it, then `think_open` (a thinking block)
/// or `answer_open` (the answer; the model may also write the answer right after the header), and `think_close` ends
/// a block. The guard lets any number of thinking blocks through and holds the grammar to the answer block, whose end
/// it allows, as an end token, only once the grammar is complete (as llama.cpp's parser for such a reply: zero or more header +
/// thinking, then header + text + the value + end).
struct ReplyForm {
    std::string think_open = "<think>", think_close = "</think>";
    std::string header, answer_open;
};

/// The running model's form (the default until a model sets its own); vocab_from_pack reads it.
void set_reply_form(const ReplyForm& form);
const ReplyForm& reply_form();

}  // namespace basalt::program::grammar
