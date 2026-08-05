"""Reconstruct a minimal HuggingFace-style ``tokenizer.json`` from a GGUF's
embedded tokenizer metadata, so the native C++ runtime can tokenize without a
separate download. The C++ ``GemmaTokenizer`` only reads ``model.vocab``,
``model.merges`` and ``added_tokens`` (it hard-codes the space→▁ normalizer and
byte-fallback), so a minimal file is sufficient and stays fully offline.
"""
import json
from pathlib import Path

# GGML token types (llama.cpp): 1 NORMAL, 2 UNKNOWN, 3 CONTROL, 4 USER_DEFINED,
# 5 UNUSED, 6 BYTE. CONTROL + USER_DEFINED are the special tokens matched atomically.
_SPECIAL_TYPES = {3, 4}


def _field_strings(reader, name):
    f = reader.fields.get(name)
    if f is None:
        return None
    return [bytes(f.parts[di]).decode("utf-8", "replace") for di in f.data]


def _field_ints(reader, name):
    f = reader.fields.get(name)
    if f is None:
        return None
    return [int(f.parts[di][0]) for di in f.data]


def _field_scalar(reader, name):
    f = reader.fields.get(name)
    if f is None or not len(f.data):
        return None
    return int(f.parts[f.data[-1]][0])


def _field_floats(reader, name):
    f = reader.fields.get(name)
    if f is None:
        return None
    return [float(f.parts[di][0]) for di in f.data]


def _reconstruct_merges_from_scores(tokens, scores):
    """Rebuild BPE merges for a SentencePiece (llama/SPM) tokenizer that ships
    unigram *scores* but no explicit merge list — e.g. the Gemma-3 GGUF. The C++
    ``GemmaTokenizer`` is merge-based, so without this it byte-fallbacks every
    word into per-character tokens (word-salad output).

    For each multi-char piece, pick the split (left+right, both in vocab) with the
    best combined score; that is the merge that produced it. Ordering merges by the
    resulting piece's score (high→low) reproduces SPM's merge priority. Matches the
    HuggingFace ``SentencePieceExtractor`` and yields correct tokenization.
    """
    vocab = {t: i for i, t in enumerate(tokens)}
    score = {t: scores[i] for i, t in enumerate(tokens)}
    merges = []
    for piece in vocab:
        if len(piece) < 2:
            continue
        best = None
        for i in range(1, len(piece)):
            left, right = piece[:i], piece[i:]
            if left in vocab and right in vocab:
                s = score[left] + score[right]
                if best is None or s > best[0]:
                    best = (s, left, right)
        if best is not None:
            merges.append((best[1], best[2], score[piece]))
    merges.sort(key=lambda x: -x[2])
    return [[a, b] for a, b, _ in merges]


def build_tokenizer_json(reader) -> dict | None:
    """Return the tokenizer.json dict, or None if the GGUF has no embedded tokens."""
    tokens = _field_strings(reader, "tokenizer.ggml.tokens")
    merges = _field_strings(reader, "tokenizer.ggml.merges") or []
    ttypes = _field_ints(reader, "tokenizer.ggml.token_type")
    if not tokens:
        return None

    vocab = {tok: i for i, tok in enumerate(tokens)}

    # Merges are stored as "left right"; tokens never contain a raw space (space
    # is the ▁ marker), so a single split recovers the pair. Order == BPE rank.
    merge_pairs = []
    for m in merges:
        left, sep, right = m.partition(" ")
        if sep:
            merge_pairs.append([left, right])

    # SentencePiece GGUFs (tokenizer.ggml.model == "llama", e.g. Gemma-3) carry
    # unigram scores but NO merges. The merge-based C++ tokenizer needs them, so
    # reconstruct from the scores (see _reconstruct_merges_from_scores).
    if not merge_pairs:
        scores = _field_floats(reader, "tokenizer.ggml.scores")
        if scores and len(scores) == len(tokens):
            merge_pairs = _reconstruct_merges_from_scores(tokens, scores)
            print(f"  (no GGUF merges — reconstructed {len(merge_pairs)} from SPM scores)")

    # Special tokens: CONTROL/USER_DEFINED plus the ids named in metadata.
    special_ids = set()
    if ttypes:
        special_ids = {i for i, t in enumerate(ttypes) if t in _SPECIAL_TYPES}
    for key in ("bos", "eos", "unknown", "padding", "mask"):
        tid = _field_scalar(reader, f"tokenizer.ggml.{key}_token_id")
        if tid is not None and 0 <= tid < len(tokens):
            special_ids.add(tid)

    added_tokens = [
        {"id": i, "content": tokens[i], "special": True}
        for i in sorted(special_ids)
    ]

    return {
        "version": "1.0",
        "added_tokens": added_tokens,
        "model": {"type": "BPE", "vocab": vocab, "merges": merge_pairs},
    }


def write_tokenizer_json(reader, out_dir) -> bool:
    """Write ``<out_dir>/tokenizer.json`` from the GGUF. Returns True on success."""
    data = build_tokenizer_json(reader)
    if data is None:
        return False
    out = Path(out_dir) / "tokenizer.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"Saved tokenizer.json: {len(data['model']['vocab'])} tokens, "
          f"{len(data['model']['merges'])} merges, "
          f"{len(data['added_tokens'])} special tokens.")
    return True
