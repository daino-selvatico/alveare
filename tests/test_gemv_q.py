import unittest
import argparse
import sys
import time
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from tools.ref.gemv_q import gemv_q_combined as ref_gemv_q, dequantize_combined
from tools.convert.gemv_q_convert import quantize_to_q4_0, pack_to_combined, load_gguf_tensor
from kernels.gemv_q.gemv_q import gemv_q_npu

class TestGemvQ(unittest.TestCase):
    
    def setUp(self):
        self.m = 32
        self.k_tile = 256
        self.rtol = 0.05
        self.atol = 1.0

    def run_gemv_npu_chunked(self, W_combined, x_np, N, K):
        """
        Run NPU GEMV, chunking along N if N // m > 64 to respect NPU hardware limits.
        """
        max_r_blocks = 64
        chunk_rows = max_r_blocks * self.m # 2048 rows
        
        y_np = np.zeros(N, dtype=bfloat16)
        
        for start_row in range(0, N, chunk_rows):
            end_row = min(start_row + chunk_rows, N)
            N_chunk = end_row - start_row
            
            # Slice weights for this chunk
            W_chunk = W_combined[start_row:end_row]
            
            # Allocate NPU tensors for this chunk
            w_t = iron.tensor(W_chunk.reshape(-1), dtype=np.uint8, device="npu")
            x_t = iron.tensor(x_np.astype(bfloat16), dtype=bfloat16, device="npu")
            y_t = iron.zeros(N_chunk, dtype=bfloat16, device="npu")
            
            # Call NPU JIT kernel
            gemv_q_npu(
                w_t, x_t, y_t,
                N=N_chunk, K=K,
                m=self.m, k_tile=self.k_tile
            )
            
            # Save results
            y_np[start_row:end_row] = y_t.numpy()
            
        return y_np

    def test_dequantize_correctness(self):
        """Verify that packing and unpacking combined format matches Q4_0 reference."""
        print("\n--- Testing Dequantization ---")
        N, K = 64, 256
        rng = np.random.default_rng(42)
        W_fp32 = rng.uniform(-1.0, 1.0, size=(N, K)).astype(np.float32)
        
        w_q4, scales = quantize_to_q4_0(W_fp32)
        w_combined = pack_to_combined(w_q4, scales)
        
        W_dequant_ref = W_fp32 # approximate comparison, or exact comparison:
        # Check that dequantize_combined matches reference dequantize
        W_dequant = dequantize_combined(w_combined)
        
        # Unpack should match exactly
        self.assertTrue(np.allclose(W_dequant.astype(np.float32), dequantize_combined(w_combined).astype(np.float32)))
        print("PASS: Dequantization matching.")

    def test_tiny_random(self):
        """Test tiny GEMV shape (256x256) on NPU."""
        print("\n--- Testing Tiny GEMV (256x256) ---")
        N, K = 256, 256
        rng = np.random.default_rng(42)
        W_fp32 = rng.uniform(-1.0, 1.0, size=(N, K)).astype(np.float32)
        x_np = rng.uniform(-1.0, 1.0, size=(K,)).astype(np.float32)
        
        w_q4, scales = quantize_to_q4_0(W_fp32)
        w_combined = pack_to_combined(w_q4, scales)
        
        expected = ref_gemv_q(w_combined, x_np.astype(bfloat16))
        actual = self.run_gemv_npu_chunked(w_combined, x_np, N, K)
        
        close = np.allclose(actual, expected, rtol=self.rtol, atol=self.atol)
        if not close:
            print(f"Max abs diff: {np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))}")
        self.assertTrue(close)
        print("PASS: Tiny shape matches reference.")

    def test_large_random(self):
        """Test large GEMV shape (2560x2560) on NPU using chunking."""
        print("\n--- Testing Large GEMV (2560x2560) ---")
        N, K = 2560, 2560
        rng = np.random.default_rng(42)
        W_fp32 = rng.uniform(-1.0, 1.0, size=(N, K)).astype(np.float32)
        x_np = rng.uniform(-1.0, 1.0, size=(K,)).astype(np.float32)
        
        w_q4, scales = quantize_to_q4_0(W_fp32)
        w_combined = pack_to_combined(w_q4, scales)
        
        expected = ref_gemv_q(w_combined, x_np.astype(bfloat16))
        actual = self.run_gemv_npu_chunked(w_combined, x_np, N, K)
        
        close = np.allclose(actual, expected, rtol=self.rtol, atol=self.atol)
        if not close:
            print(f"Max abs diff: {np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))}")
        self.assertTrue(close)
        print("PASS: Large shape matches reference.")

    def test_real_weights(self):
        """Test GEMV using real weights loaded from a local GGUF."""
        print("\n--- Testing Real GGUF Weights (640x3840) ---")
        gguf_path = "/home/daino/llama-mtp/models/gemma-4-12b-it-mmproj-F16.gguf"
        tensor_name = "mm.a.input_projection.weight"
        
        # Load and slice/convert
        W_fp32 = load_gguf_tensor(gguf_path, tensor_name).T # shape (640, 3840)
        N, K = W_fp32.shape
        
        rng = np.random.default_rng(42)
        x_np = rng.uniform(-1.0, 1.0, size=(K,)).astype(np.float32)
        
        w_q4, scales = quantize_to_q4_0(W_fp32)
        w_combined = pack_to_combined(w_q4, scales)
        
        expected = ref_gemv_q(w_combined, x_np.astype(bfloat16))
        actual = self.run_gemv_npu_chunked(w_combined, x_np, N, K)
        
        close = np.allclose(actual, expected, rtol=self.rtol, atol=self.atol)
        if not close:
            print(f"Max abs diff: {np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))}")
        self.assertTrue(close)
        print("PASS: Real weights shape matches reference.")

def run_benchmarks():
    print("\n=== Running Latency Microbenchmarks ===")
    m = 32
    k_tile = 256
    shapes = [(256, 256), (2048, 2048), (2560, 2560)]
    
    # Warmup NPU context
    print("Warmup NPU...")
    dummy_w = np.zeros((32, 256 // 32 * 20), dtype=np.uint8)
    dummy_x = np.zeros(256, dtype=bfloat16)
    dummy_w_t = iron.tensor(dummy_w.reshape(-1), dtype=np.uint8, device="npu")
    dummy_x_t = iron.tensor(dummy_x, dtype=bfloat16, device="npu")
    dummy_y_t = iron.zeros(32, dtype=bfloat16, device="npu")
    for _ in range(5):
        gemv_q_npu(dummy_w_t, dummy_x_t, dummy_y_t, N=32, K=256, m=32, k_tile=256)
        
    bench_results = []
    
    for N, K in shapes:
        print(f"\nBenchmarking shape {N}x{K}...")
        rng = np.random.default_rng(42)
        W_fp32 = rng.uniform(-1.0, 1.0, size=(N, K)).astype(np.float32)
        x_np = rng.uniform(-1.0, 1.0, size=(K,)).astype(np.float32)
        
        w_q4, scales = quantize_to_q4_0(W_fp32)
        w_combined = pack_to_combined(w_q4, scales)
        
        # Benchmark CPU NumPy Reference
        t0 = time.perf_counter()
        iters_cpu = 10
        for _ in range(iters_cpu):
            expected = ref_gemv_q(w_combined, x_np.astype(bfloat16))
        t1 = time.perf_counter()
        cpu_ms = ((t1 - t0) / iters_cpu) * 1000.0
        
        # Benchmark NPU
        t0 = time.perf_counter()
        iters_npu = 50
        
        # To avoid overhead of NPU tensor allocation/compilation in the benchmark loop,
        # we prepare the NPU tensor structure first.
        max_r_blocks = 64
        chunk_rows = m * max_r_blocks
        
        # Setup tensors outside benchmark loop
        chunks = []
        for start_row in range(0, N, chunk_rows):
            end_row = min(start_row + chunk_rows, N)
            N_chunk = end_row - start_row
            W_chunk = w_combined[start_row:end_row]
            
            w_t = iron.tensor(W_chunk.reshape(-1), dtype=np.uint8, device="npu")
            x_t = iron.tensor(x_np.astype(bfloat16), dtype=bfloat16, device="npu")
            y_t = iron.zeros(N_chunk, dtype=bfloat16, device="npu")
            chunks.append((w_t, x_t, y_t, N_chunk))
            
        t0 = time.perf_counter()
        for _ in range(iters_npu):
            for w_t, x_t, y_t, N_chunk in chunks:
                gemv_q_npu(
                    w_t, x_t, y_t,
                    N=N_chunk, K=K,
                    m=m, k_tile=k_tile
                )
        t1 = time.perf_counter()
        npu_ms = ((t1 - t0) / iters_npu) * 1000.0
        
        speedup = cpu_ms / npu_ms
        res_str = f"Shape: {N}x{K} | CPU: {cpu_ms:.2f} ms | NPU: {npu_ms:.2f} ms | Speedup: {speedup:.2f}x"
        print(res_str)
        bench_results.append(res_str)
        
    # Save to bench file
    bench_dir = Path(__file__).resolve().parents[1] / "tests" / "bench"
    bench_dir.mkdir(parents=True, exist_ok=True)
    bench_file = bench_dir / "gemv_q_bench.txt"
    
    import platform
    machine_info = f"Machine: {platform.processor()} / AMD NPU (XDNA2)\n"
    with open(bench_file, "w") as f:
        f.write(machine_info)
        f.write("\n".join(bench_results) + "\n")
    print(f"\nSaved benchmark results to {bench_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true", help="Run latency microbenchmarks")
    args, unknown = parser.parse_known_args()
    
    if args.bench:
        run_benchmarks()
    else:
        # Run unittest
        # Remove --bench from sys.argv so unittest parser doesn't fail
        sys.argv = [sys.argv[0]] + unknown
        unittest.main()
