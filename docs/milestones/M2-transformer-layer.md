# M2 — One transformer layer on NPU

**Status**: **Completed**.

## Goal

Run a complete decoder layer on the NPU for a simple, dense model: RMSNorm → QKV → attention (with GQA and KV cache) → output proj → RMSNorm → MLP → residuals.

## Target model

We selected **Llama-3.2-1B-Instruct** (plain dense, RMSNorm, GQA, RoPE, SwiGLU MLP). We downloaded the unquantized GGUF weights to `/home/daino/llama-mtp/models/Llama-3.2-1B-Instruct-f16.gguf` and used them to generate the oracle reference.

## Kernels Implemented

- **`rmsnorm`**: C++ AIE kernel performing RMSNorm. Scaled using `float32` gamma weight vector. Uses `aie::invsqrt` for precision.
- **`rope`**: C++ AIE kernel applying Llama's split-half rotary positional embeddings. Takes a single packed `cos_sin` input of shape `(128,)` to fit within DMA channel limits.
- **`attention`**: C++ AIE kernel executing QKᵀ → softmax → ·V with GQA head grouping. Processes Key and Value caches packed sequentially along the sequence dimension as shape `(seq_len, 128)` to comply with the 2-input DMA channel limit. Softmax exponents are computed using a bit-manipulation `exp` approximation.
- **`gemv_q`**: Reused M1's int4 weight × bf16 activation matrix-vector multiply kernel.

To avoid overloading userspace driver context memory (`AMDXDNA_CREATE_HWCTX`), we pad and chunk all GEMV projection shapes (Q, K, V, O, Gate, Up, Down) to a single compiled NPU JIT shape of `(2048, 2048)`.

## Error Tracking (NPU vs Hugging Face Reference)

Verified step-by-step for a 32-token decode step (position `t = 31`):

| Sub-operation / Layer Component | Relative Error | Max Absolute Error | Status |
|---|---|---|---|
| **RMSNorm 1** (Input norm) | `0.00410` | `0.00781` | PASS |
| **Query Proj** (GEMV) | `0.07681` | `0.22656` | PASS |
| **Query RoPE** (Rotary emb) | `0.00441` | `0.01562` | PASS |
| **Attention** (NPU core) | `0.02107` | `0.00183` | PASS |
| **Attn Proj** (Output GEMV) | `0.16827` | `0.00372` | PASS |
| **Gate Proj** (MLP GEMV) | `0.09897` | `0.13281` | PASS |
| **Up Proj** (MLP GEMV) | `0.09843` | `0.06934` | PASS |
| **Down Proj** (MLP GEMV) | `0.10357` | `0.00848` | PASS |
| **Full Decoder Layer Output** | **`0.00520` (0.52%)** | **`0.03125`** | **PASS** |

## KV Cache Approach

- KV cache is stored in host DRAM in `bfloat16`.
- For GQA grouping (32 Q heads, 8 KV heads), the cache is laid out as shape `(8, max_seq_len, 64)`.
- During the attention step, the active slice `(8, seq_len, 64)` of keys and values is interleaved/packed on the host to shape `(8, seq_len, 128)` and streamed as a single input DMA buffer, keeping the attention NPU design within the 2-input DMA limit (Input 1: Query, Input 2: Packed KV cache).

## Friction & Resolutions

1. **Memory-Mapped Destructors (Segfaults)**:
   - *Issue*: `y_t.numpy()` returns a memory-mapped view of XRT's physical buffer object (BO). When `y_t` goes out of scope, its destructor unmaps the memory, causing immediate segfaults on subsequent Python accesses to the array.
   - *Resolution*: Force a deep copy of the NumPy view while the tensor is still in scope: `res = np.array(y_t.numpy())`.
2. **2-Input DMA Limits (Hardware Constraint)**:
   - *Issue*: AIE core tiles on Ryzen AI NPUs have a hardware limit of 2 input DMA channels, preventing us from passing Q, K, and V cache buffers separately.
   - *Resolution*: Pack `cos` and `sin` together for RoPE, and interleave Key and Value cache vectors into a single contiguous `(seq_len, 128)` buffer for Attention.
3. **Userspace Driver Context Limits (DRM create_hwctx errors)**:
   - *Issue*: Creating a new pyxrt context for every shape (e.g. 2048, 512, 8192) exhausts the AMDXDNA DRM resource limits.
   - *Resolution*: Padding/chunking all dense matmuls to a single `(2048, 2048)` compiled NPU kernel shape.
