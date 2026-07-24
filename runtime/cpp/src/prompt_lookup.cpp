#include "alveare/prompt_lookup.h"

#include <algorithm>

namespace alveare {

std::vector<int> propose_draft(const std::vector<int>& tokens,
                               int max_draft,
                               int max_ngram,
                               int min_ngram) {
    const int n = static_cast<int>(tokens.size());
    if (n <= 1 || max_draft <= 0 || max_ngram < 1 || min_ngram < 1 || min_ngram > max_ngram) {
        return {};
    }

    for (int g = max_ngram; g >= min_ngram; --g) {
        if (g > n - 1) {
            continue;
        }

        // Search for the largest index i in [0, n - g - 1] such that
        // tokens[i .. i + g - 1] == pattern tokens[n - g .. n - 1].
        for (int i = n - g - 1; i >= 0; --i) {
            bool match = true;
            for (int k = 0; k < g; ++k) {
                if (tokens[i + k] != tokens[n - g + k]) {
                    match = false;
                    break;
                }
            }

            if (match) {
                int start = i + g;
                int end = std::min(n - g, start + max_draft);
                if (start >= end) {
                    return {};
                }
                return std::vector<int>(tokens.begin() + start, tokens.begin() + end);
            }
        }
    }

    return {};
}

} // namespace alveare
