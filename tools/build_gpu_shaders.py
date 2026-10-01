#!/usr/bin/env python3
"""
Alveare GPU Shader Compiler.
Compiles GLSL compute shaders in kernels/gpu/ to SPIR-V binaries (.spv)
using glslc or glslangValidator, and regenerates runtime/cpp/include/alveare/spv_shaders.h.
"""
import os
import sys
import shutil
import struct
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GPU_DIR = ROOT / "kernels" / "gpu"
HEADER_PATH = ROOT / "runtime" / "cpp" / "include" / "alveare" / "spv_shaders.h"

def find_compiler():
    if shutil.which("glslc"):
        return "glslc"
    if shutil.which("glslangValidator"):
        return "glslangValidator"
    return None

def compile_shader(compiler: str, comp_file: Path, spv_file: Path):
    if compiler == "glslc":
        cmd = ["glslc", "--target-env=vulkan1.3", "-fshader-stage=compute", str(comp_file), "-o", str(spv_file)]
    else:
        cmd = ["glslangValidator", "--target-env", "vulkan1.3", "-V", str(comp_file), "-o", str(spv_file)]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Shader compilation failed for {comp_file.name}:\n{res.stderr or res.stdout}")
    print(f"✓ Compiled {comp_file.name} -> {spv_file.name} ({spv_file.stat().st_size} bytes)")

def spv_to_uint32_array(spv_path: Path) -> list:
    data = spv_path.read_bytes()
    if len(data) % 4 != 0:
        raise ValueError(f"SPIR-V size ({len(data)}) is not a multiple of 4")
    words = struct.unpack(f"<{len(data) // 4}I", data)
    return words

def generate_header(gemv_words: list, gemm_words: list, out_path: Path):
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"✓ Generated C++ shader header: {out_path} ({len(gemv_words)} words GEMV, {len(gemm_words)} words GEMM)")

def rebuild_cpp_runtime():
    build_dir = ROOT / "runtime" / "cpp" / "build"
    if build_dir.exists() and (build_dir / "Makefile").exists() and shutil.which("cmake"):
        print("Recompiling C++ runtime with updated GPU shaders...")
        res = subprocess.run(["cmake", "--build", str(build_dir), "-j"], capture_output=True, text=True)
        if res.returncode == 0:
            print("✓ C++ runtime successfully recompiled with new SPIR-V kernels.")
        else:
            print(f"warning: C++ runtime build notice:\n{res.stderr or res.stdout}")

def main():
    compiler = find_compiler()
    if not compiler:
        print("error: Neither glslc nor glslangValidator found in PATH.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Using shader compiler: {compiler}")
    gemv_comp = GPU_DIR / "gemv_q4_0.comp"
    gemm_comp = GPU_DIR / "gemm_q4_0.comp"
    gemv_spv = GPU_DIR / "gemv_q4_0.spv"
    gemm_spv = GPU_DIR / "gemm_q4_0.spv"
    
    compile_shader(compiler, gemv_comp, gemv_spv)
    compile_shader(compiler, gemm_comp, gemm_spv)
    
    gemv_words = spv_to_uint32_array(gemv_spv)
    gemm_words = spv_to_uint32_array(gemm_spv)
    generate_header(gemv_words, gemm_words, HEADER_PATH)
    rebuild_cpp_runtime()
    print("GPU SPIR-V shader build complete.")

if __name__ == "__main__":
    main()
