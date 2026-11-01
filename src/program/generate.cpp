// src/program/generate.cpp - the generate driver: it composes the components into a token.
//
//     embed_row(token)  ->  48 captured layer graphs (the CPU expert pool behind the doorbell)  ->
//     lm_head(R)        ->  sample        ->  embed_row(next)  ->  ...
//
// There is no tokenizer here: the prompt arrives as ids via `--tokens`, so a logit comparison against a reference runs
// on the same ids on both sides.

#include "generate/run.hpp"

int main(int argc, char** argv) {
    return basalt::program::generate::run(argc, argv);
}
