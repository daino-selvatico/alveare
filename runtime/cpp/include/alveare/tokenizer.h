#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace alveare {

class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    virtual std::vector<int> encode(const std::string& text) const = 0;
    virtual std::string decode(const std::vector<int>& tokens) const = 0;
    virtual std::string decode(int token) const = 0;
    virtual int bos_token_id() const = 0;
    virtual int eos_token_id() const = 0;

    // True if generation should stop after emitting this token. Defaults to EOS.
    virtual bool is_stop_token(int token) const { return token == eos_token_id(); }
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

// Byte-level BPE tokenizer that loads a HuggingFace `tokenizer.json` (the format
// used by Gemma / Llama SentencePiece-BPE exports). Implements the subset the
// Gemma models need: a Replace(" " -> "▁") normalizer, rank-ordered BPE over
// UTF-8 characters, byte fallback for out-of-vocab characters, and atomic matching
// of the special/added tokens. Throws std::runtime_error if the file can't load.
class GemmaTokenizer : public Tokenizer {
public:
    explicit GemmaTokenizer(const std::string& tokenizer_json_path);

    std::vector<int> encode(const std::string& text) const override;
    std::string decode(const std::vector<int>& tokens) const override;
    std::string decode(int token) const override;
    int bos_token_id() const override { return bos_id_; }
    int eos_token_id() const override { return eos_id_; }
    bool is_stop_token(int token) const override;

private:
    std::unordered_map<std::string, int> vocab_;      // piece -> id
    std::vector<std::string> id_to_token_;            // id -> piece
    std::unordered_map<std::string, int> merge_ranks_; // "left\0right" -> rank
    std::vector<char> byte_token_id_;                 // byte value -> valid?
    std::vector<int> byte_to_id_;                     // byte value -> <0xNN> id

    // Special (added) tokens, sorted by content length desc for longest-match.
    struct Special { std::string content; int id; };
    std::vector<Special> specials_;
    std::vector<char> is_special_id_;                 // id -> is special

    int bos_id_ = 2;
    int eos_id_ = 1;
    int turn_end_id_ = -1;

    void bpe_segment(const std::string& normalized, std::vector<int>& out) const;
};

} // namespace alveare
