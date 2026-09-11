#!/usr/bin/env bash
# ==============================================================================
# Alveare GPU Shader Build Script
# Deterministically compiles GLSL compute shaders into SPIR-V binaries (.spv)
# and generates the embedded C++ header runtime/cpp/include/alveare/spv_shaders.h
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GPU_DIR="${ROOT_DIR}/kernels/gpu"
HEADER_PATH="${ROOT_DIR}/runtime/cpp/include/alveare/spv_shaders.h"

COMPILER="${GLSLC:-$(command -v glslc || true)}"
if [[ -z "${COMPILER}" ]]; then
    echo "ERROR: glslc not found in PATH. Please install shaderc/vulkan-tools or set GLSLC." >&2
    exit 1
fi

echo "==> Using shader compiler: ${COMPILER}"

TARGET_ENV="vulkan1.2"

echo "==> Compiling gemv_q4_0.comp..."
"${COMPILER}" "--target-env=${TARGET_ENV}" -O -fshader-stage=compute \
    "${GPU_DIR}/gemv_q4_0.comp" -o "${GPU_DIR}/gemv_q4_0.spv"

echo "==> Compiling gemm_q4_0.comp..."
"${COMPILER}" "--target-env=${TARGET_ENV}" -O -fshader-stage=compute \
    "${GPU_DIR}/gemm_q4_0.comp" -o "${GPU_DIR}/gemm_q4_0.spv"

echo "==> Regenerating C++ header: ${HEADER_PATH}..."
python3 - <<EOF
import struct
from pathlib import Path

header_path = Path("${HEADER_PATH}")
gemv_spv = Path("${GPU_DIR}/gemv_q4_0.spv")
gemm_spv = Path("${GPU_DIR}/gemm_q4_0.spv")

def to_words(spv_path):
    data = spv_path.read_bytes()
    assert len(data) % 4 == 0, f"Unaligned SPIR-V in {spv_path}"
    return struct.unpack(f"<{len(data)//4}I", data)

gemv_words = to_words(gemv_spv)
gemm_words = to_words(gemm_spv)

lines = [
    "#pragma once",
    "#include <cstdint>",
    "#include <cstddef>",
    "",
    "namespace alveare {",
    "namespace gpu_shaders {",
    "inline const uint32_t gemv_q4_0_spv[] = {"
]
for i in range(0, len(gemv_words), 8):
    chunk = gemv_words[i:i+8]
    hex_str = ", ".join(f"0x{w:08x}u" for w in chunk) + ", "
    lines.append(hex_str)
lines.extend([
    "};",
    "inline const size_t gemv_q4_0_spv_size = sizeof(gemv_q4_0_spv);",
    "",
    "inline const uint32_t gemm_q4_0_spv[] = {"
])
for i in range(0, len(gemm_words), 8):
    chunk = gemm_words[i:i+8]
    hex_str = ", ".join(f"0x{w:08x}u" for w in chunk) + ", "
    lines.append(hex_str)
lines.extend([
    "};",
    "inline const size_t gemm_q4_0_spv_size = sizeof(gemm_q4_0_spv);",
    "",
    "} // namespace gpu_shaders",
    "} // namespace alveare",
    ""
])
header_path.write_text("\n".join(lines))
print(f"Header successfully written ({len(gemv_words)} words GEMV, {len(gemm_words)} words GEMM)")
EOF

echo "==> Build Identity & Checksums:"
sha256sum "${GPU_DIR}/gemv_q4_0.comp" \
          "${GPU_DIR}/gemv_q4_0.spv" \
          "${GPU_DIR}/gemm_q4_0.comp" \
          "${GPU_DIR}/gemm_q4_0.spv" \
          "${HEADER_PATH}"

echo "==> Shader build completed successfully."
