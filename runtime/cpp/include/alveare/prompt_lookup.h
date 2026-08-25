#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>

namespace alveare {

// Proposes up to max_draft candidate tokens by matching the trailing n-gram
// pattern in the sequence of tokens using prompt-lookup (n-gram) decoding.
// Searches for the most recent occurrence (scanned from right, excluding the current suffix)
// of an n-gram matching the tail of `tokens`, and returns the tokens following that occurrence.
// Tries n-gram lengths from `max_ngram` down to `min_ngram`, returning early on the first match.
// Returns an empty vector if no match is found.
std::vector<int> propose_draft(const std::vector<int>& tokens,
                               int max_draft = 4,
                               int max_ngram = 3,
                               int min_ngram = 2);

class DynamicDrafter {
public:
    DynamicDrafter();
    ~DynamicDrafter();
    void reset();
    void feed_sequence(const std::vector<int>& tokens);
    void feed_token(int t);
    std::vector<int> draft(const std::vector<int>& context, int max_draft = 7);

private:
    std::vector<int> history_;
    std::vector<uint64_t> bigram_keys_;
    std::vector<int> bigram_vals_;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace alveare
