#include "alveare/tokenizer.h"

namespace alveare {

std::vector<int> StubTokenizer::encode(const std::string& text) const {
    std::vector<int> tokens;
    // For stub, cast char to unsigned char first to avoid negative tokens
    for (char c : text) {
        tokens.push_back(static_cast<int>(static_cast<unsigned char>(c)));
    }
    return tokens;
}

std::string StubTokenizer::decode(const std::vector<int>& tokens) const {
    std::string text;
    for (int t : tokens) {
        text += decode(t);
    }
    return text;
}

std::string StubTokenizer::decode(int token) const {
    if (token == bos_token_id() || token == eos_token_id()) {
        return "";
    }
    // Assume ASCII
    char c = static_cast<char>(token);
    return std::string(1, c);
}

} // namespace alveare
