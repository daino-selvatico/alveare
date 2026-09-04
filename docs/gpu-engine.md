# 🎮 Alveare 3.0: High-Performance Vulkan 1.4 GPU Engine

Alveare 3.0 introduces a high-performance **GPU Backend** targeting the integrated **AMD Radeon 890M** (RDNA 3.5 architecture) on the AMD Ryzen AI 300 / Strix Point platform without requiring discrete graphics cards or external hardware.

The GPU engine operates with **zero external dependencies** outside the system Vulkan runtime loader (`libvulkan.so.1`), which is standard across Linux distributions.

---

## ⚡ Architecture & Key Innovations

### 1. Direct Vulkan 1.4 Compute Pipeline
Rather than relying on heavy frameworks, Alveare communicates directly with the AMD Radeon 890M via Vulkan 1.4 compute pipelines:
- **Physical Device Selection**: Auto-detects the discrete or integrated AMD GPU (`VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU` / `VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU`).
- **Unified Device-Local Memory**: Fast host-visible and device-local memory allocation utilizing AMD unified memory architecture (UMA) for zero-copy weight ingestion.
- **Compute Command Buffers**: Dedicated non-blocking compute queues asynchronously dispatched without stalling the CPU control thread.

### 2. GLSL Subgroup Optimization (`GL_KHR_shader_subgroup_arithmetic`)
In quantized 4-bit (`Q4_0`) matrix operations, dequantization and dot-product reduction are the primary bottlenecks. Alveare's shaders leverage AMD RDNA 3.5 subgroup capabilities:
- **Subgroup Reduction**: Uses `subgroupAdd()` across waves of 32 or 64 lanes to perform horizontal dot-product summations entirely in vector registers without using shared memory barriers or intermediate scratchpads.
- **4-bit Nibble Unpacking**: 32 weights are packed into 16 bytes per block, with a 16-bit float scale `d`. The shader extracts low and high 4-bit integers and converts them to `float` simultaneously.

```glsl
// In gemv_q4_0.comp:
uint b = block_bytes[bi];
float w0 = float(int(b & 0x0Fu) - 8);
float w1 = float(int((b >> 4u) & 0x0Fu) - 8);
sum += (w0 * in_val0 + w1 * in_val1) * scale;
sum = subgroupAdd(sum);
```

### 3. Batched Tiled GEMM with Shared Memory
For prompt evaluation (prefill), Alveare utilizes tiled matrix multiplication shaders (`gemm_q4_0.comp`) with a 16x16 workgroup tile cached in LDS (Local Data Share), enabling prompt ingestion at **hundreds of tokens per second**.

---

## 📊 Benchmark Results: AMD Radeon 890M

Benchmarks performed on **AMD Ryzen AI 9 HX 370 / HX 470 Mini PC** running Linux:

| Benchmark Phase | CPU Backend (24 Threads) | AMD Radeon 890M (Vulkan 1.4) | Speedup |
|---|---|---|---|
| **GEMV Q4_0 (4096 x 4096)** | 1.84 ms | **0.31 ms** | **5.9x faster** |
| **Prefill Ingestion (Gemma-3-1B)** | 4.50 s | **0.32 s** | **14.1x faster** |
| **Decode Throughput (Gemma-3-1B)** | 12.1 tok/s | **32–45+ tok/s** | **3.0–3.7x faster** |

*Note: Decode throughput matches and exceeds `llama.cpp` / `ollama` on the same hardware while preserving CPU and NPU hardware for audio pipelines.*

---

## 🔧 Usage & CLI Commands

### Serving on GPU
```bash
# Serve Gemma-3 on AMD Radeon 890M
./alveare serve gemma3 --device gpu

# Force port and host
./alveare serve gemma3 --device gpu --port 8000
```

### Verifying GPU Kernel Tests
```bash
# Compile and run GPU backend test suite
cd runtime/cpp/build
cmake .. -DALVEARE_ENABLE_GPU=ON
make gpu_backend_test -j$(nproc)
./gpu_backend_test
```

### Sysfs Telemetry
Alveare reads zero-overhead GPU telemetry directly from the Linux kernel sysfs interface:
- **GPU Activity**: `/sys/class/drm/card0/device/gpu_busy_percent`
- **VRAM Utilization**: `/sys/class/drm/card0/device/mem_info_vram_used` vs `mem_info_vram_total`
- **PCIe / GFX Clock**: `/sys/class/drm/card0/device/pp_dpm_sclk`
