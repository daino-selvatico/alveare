#!/usr/bin/env python3
"""Detect a GGUF's model architecture and map it to Alveare's quantizer key.

Reads `general.architecture` from the GGUF metadata and maps it to one of the
architectures Alveare supports (llama / gemma3 / gemma4). Prints the key on
success; exits non-zero with a message on an unsupported architecture.

Same Q4_0 quantization *algorithm* for all; the architecture only selects the
per-model tensor wiring (which quantizer script + config to produce).

Usage:  python tools/detect_arch.py <model.gguf>
"""
import sys

try:
    from gguf import GGUFReader
except ModuleNotFoundError:
    import os
    sys.path.append(os.environ.get("LLAMA_CPP_GGUF_PY", "/home/daino/llama-mtp/llama.cpp/gguf-py"))
    from gguf import GGUFReader

# GGUF `general.architecture` string -> Alveare quantizer key.
SUPPORTED = {
    "llama": "llama",
    "gemma3": "gemma3",
    "gemma4": "gemma4",
}


def read_architecture(path):
    reader = GGUFReader(path)
    field = reader.fields.get("general.architecture")
    if field is None:
        return None
    # Robustly extract the string value across gguf versions.
    try:
        return str(bytes(field.parts[field.data[0]]), "utf-8")
    except Exception:
        try:
            return field.contents()
        except Exception:
            return None


def main():
    if len(sys.argv) != 2:
        print("usage: detect_arch.py <model.gguf>", file=sys.stderr)
        sys.exit(2)
    gguf = sys.argv[1]
    try:
        arch = read_architecture(gguf)
    except Exception as e:
        print(f"error: could not read GGUF metadata from {gguf}: {e}", file=sys.stderr)
        sys.exit(1)

    if arch is None:
        print(f"error: no 'general.architecture' field in {gguf}", file=sys.stderr)
        sys.exit(1)

    key = SUPPORTED.get(arch)
    if key is None:
        print(
            f"error: unsupported architecture '{arch}'. "
            f"Alveare currently supports: {', '.join(sorted(set(SUPPORTED.values())))}.\n"
            f"Adding a new architecture needs a new quantizer + runtime support "
            f"(see the milestone docs).",
            file=sys.stderr,
        )
        sys.exit(3)

    print(key)


if __name__ == "__main__":
    main()
