#include "alveare/tokenizer.h"
#include "nlohmann/json.hpp"

#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <climits>
#include <cstdio>

using json = nlohmann::json;

namespace alveare {

// UTF-8 marker "▁" (U+2581) that the normalizer substitutes for a space.
static const std::string kSpaceMarker = "\xe2\x96\x81";

static int utf8_char_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c >> 5) == 0x6) return 2;
    if ((c >> 4) == 0xE) return 3;
    if ((c >> 3) == 0x1E) return 4;
    return 1; // invalid lead byte: treat as a single byte
}

static int hex_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

GemmaTokenizer::GemmaTokenizer(const std::string& tokenizer_json_path) {
    std::ifstream f(tokenizer_json_path);
    if (!f) throw std::runtime_error("tokenizer: cannot open " + tokenizer_json_path);

    json d;
    f >> d;

    const json& model = d.at("model");

    // Vocabulary: piece -> id.
    const json& vocab = model.at("vocab");
    vocab_.reserve(vocab.size() * 2);
    int max_id = 0;
    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
        int id = it.value().get<int>();
        vocab_.emplace(it.key(), id);
        if (id > max_id) max_id = id;
    }
    id_to_token_.assign(max_id + 1, std::string());
    for (const auto& kv : vocab_) id_to_token_[kv.second] = kv.first;

    // Merge ranks: pair (left,right) -> priority (lower merges first).
    const json& merges = model.at("merges");
    merge_ranks_.reserve(merges.size() * 2);
    for (size_t r = 0; r < merges.size(); ++r) {
        const json& m = merges[r];
        std::string left = m[0].get<std::string>();
        std::string right = m[1].get<std::string>();
        std::string key = left;
        key.push_back('\0');
        key += right;
        merge_ranks_.emplace(std::move(key), static_cast<int>(r));
    }

    // Byte-fallback tokens "<0xNN>".
    byte_to_id_.assign(256, -1);
    for (int b = 0; b < 256; ++b) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
        auto it = vocab_.find(buf);
        if (it != vocab_.end()) byte_to_id_[b] = it->second;
    }

    // Special / added tokens, longest content first for greedy matching.
    is_special_id_.assign(max_id + 1, 0);
    int turn_end_id = -1;
    if (d.contains("added_tokens")) {
        for (const json& t : d.at("added_tokens")) {
            Special sp{t.at("content").get<std::string>(), t.at("id").get<int>()};
            if (sp.id >= 0 && sp.id <= max_id) {
                is_special_id_[sp.id] = 1;
                id_to_token_[sp.id] = sp.content;
            }
            if (sp.content == "<bos>") bos_id_ = sp.id;
            else if (sp.content == "<eos>") eos_id_ = sp.id;
            else if (sp.content == "<turn|>") turn_end_id = sp.id;
            specials_.push_back(std::move(sp));
        }
    }
    std::sort(specials_.begin(), specials_.end(),
              [](const Special& a, const Special& b) { return a.content.size() > b.content.size(); });
    turn_end_id_ = turn_end_id;
}

void GemmaTokenizer::bpe_segment(const std::string& s, std::vector<int>& out) const {
    if (s.empty()) return;

    // Split into UTF-8 characters.
    std::vector<std::string> syms;
    for (size_t i = 0; i < s.size();) {
        int len = utf8_char_len(static_cast<unsigned char>(s[i]));
        if (i + len > s.size()) len = 1;
        syms.push_back(s.substr(i, len));
        i += len;
    }

    // Greedily merge the adjacent pair with the lowest rank until none apply.
    while (syms.size() > 1) {
        int best_rank = INT_MAX;
        int best_i = -1;
        std::string key;
        for (size_t i = 0; i + 1 < syms.size(); ++i) {
            key = syms[i];
            key.push_back('\0');
            key += syms[i + 1];
            auto it = merge_ranks_.find(key);
            if (it != merge_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_i = static_cast<int>(i);
            }
        }
        if (best_i < 0) break;
        syms[best_i] += syms[best_i + 1];
        syms.erase(syms.begin() + best_i + 1);
    }

    // Map symbols to ids, falling back to per-byte tokens when out of vocab.
    for (const std::string& sym : syms) {
        auto it = vocab_.find(sym);
        if (it != vocab_.end()) {
            out.push_back(it->second);
        } else {
            for (unsigned char b : sym) {
                int id = byte_to_id_[b];
                if (id >= 0) out.push_back(id);
            }
        }
    }
}

std::vector<int> GemmaTokenizer::encode(const std::string& text) const {
    std::vector<int> out;
    std::string pending; // raw text awaiting normalization + BPE

    auto flush = [&]() {
        if (pending.empty()) return;
        std::string norm;
        norm.reserve(pending.size() + 8);
        for (char c : pending) {
            if (c == ' ') norm += kSpaceMarker;
            else norm.push_back(c);
        }
        bpe_segment(norm, out);
        pending.clear();
    };

    size_t i = 0, n = text.size();
    while (i < n) {
        int matched_len = 0, matched_id = -1;
        for (const Special& sp : specials_) { // longest first
            const std::string& c = sp.content;
            if (!c.empty() && c.size() <= n - i && text.compare(i, c.size(), c) == 0) {
                matched_len = static_cast<int>(c.size());
                matched_id = sp.id;
                break;
            }
        }
        if (matched_id >= 0) {
            flush();
            out.push_back(matched_id);
            i += matched_len;
        } else {
            pending.push_back(text[i]);
            ++i;
        }
    }
    flush();
    return out;
}

std::string GemmaTokenizer::decode(int token) const {
    if (token < 0 || token >= static_cast<int>(id_to_token_.size())) return "";
    if (is_special_id_[token]) return ""; // never render special tokens
    const std::string& t = id_to_token_[token];

    // Byte-fallback token "<0xNN>" -> raw byte.
    if (t.size() == 6 && t[0] == '<' && t[1] == '0' && t[2] == 'x' && t[5] == '>') {
        int hi = hex_val(t[3]), lo = hex_val(t[4]);
        if (hi >= 0 && lo >= 0) return std::string(1, static_cast<char>((hi << 4) | lo));
    }

    // Replace the space marker "▁" with a real space.
    std::string outp;
    outp.reserve(t.size());
    for (size_t i = 0; i < t.size();) {
        if (i + 3 <= t.size() && t.compare(i, 3, kSpaceMarker) == 0) {
            outp.push_back(' ');
            i += 3;
        } else {
            outp.push_back(t[i]);
            ++i;
        }
    }
    return outp;
}

std::string GemmaTokenizer::decode(const std::vector<int>& tokens) const {
    std::string out;
    for (int t : tokens) out += decode(t);
    return out;
}

bool GemmaTokenizer::is_stop_token(int token) const {
    return token == eos_id_ || (turn_end_id_ >= 0 && token == turn_end_id_);
}

} // namespace alveare
