#pragma once
#include <string>
#include <vector>

namespace alveare {

class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    virtual std::vector<int> encode(const std::string& text) const = 0;
    virtual std::string decode(const std::vector<int>& tokens) const = 0;
    virtual std::string decode(int token) const = 0;
    virtual int bos_token_id() const = 0;
    virtual int eos_token_id() const = 0;
};

// A dummy stub tokenizer for initial testing
class StubTokenizer : public Tokenizer {
public:
    StubTokenizer() {}

    std::vector<int> encode(const std::string& text) const override;
    std::string decode(const std::vector<int>& tokens) const override;
    std::string decode(int token) const override;
    int bos_token_id() const override { return 1; }
    int eos_token_id() const override { return 2; }
};

} // namespace alveare
