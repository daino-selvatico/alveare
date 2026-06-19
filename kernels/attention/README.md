# Attention Kernel

This kernel performs Grouped Query Attention (GQA) with a causal KV cache on the AMD Ryzen AI NPU.

## Design

For each Grouped Query Attention (GQA) head group (consisting of 4 Query heads mapped to 1 Key/Value head), the kernel computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

### Implementation Details
* **Grouped Query Attention (GQA)**: Handles 4 Query heads per single Key/Value head group. Under Llama 3.2-1B, there are 32 Query heads and 8 KV heads, resulting in 8 GQA head groups. The host invokes the kernel for each group sequentially.
* **2-Input DMA Limit Resolution**: Ryzen AI NPU core tiles support at most 2 input DMA channels. To stream Query, Key, and Value tensors, the Key and Value cache slices are packed and interleaved in host DRAM into a single contiguous buffer `kv_group` of shape `(seq_len, 2 * H)`. The first `H` elements in the inner dimension contain the Key vector, and the last `H` elements contain the Value vector.
* **Stable Softmax**: To avoid overflow/underflow, stable softmax subtraction (subtracting $max(score)$) is performed.
* **Fast Exponential on NPU**: Since standard transcendental math functions are expensive or unavailable on AIE vectors, exponential values are computed using a fast bit-manipulation integer bit-shift:
  
  $$2^{ix} \approx 1 + 0.693147 \cdot fx + 0.240160 \cdot fx^2$$

  where $ix$ is the integer part and $fx$ is the fractional part of $x \log_2(e)$.
* **Precision**: Dot product and softmax accumulation are done in `float32` to avoid compounding errors across sequence lengths.

## Host ABI

```c
void attention(
    const bfloat16 *restrict q_group,
    const bfloat16 *restrict kv_group,
    bfloat16 *restrict o_group
);
```

## Shapes and Data Types

| Argument | Description | Shape | Data Type |
| :--- | :--- | :--- | :--- |
| `q_group` | 4 Query heads in a GQA head group | `(4 * H,)` | `bfloat16` |
| `kv_group` | Packed Key and Value cache slice up to `seq_len` | `(seq_len, 2 * H)` | `bfloat16` |
| `o_group` | Output attention head group | `(4 * H,)` | `bfloat16` |

*Default `H = 64` (head dimension). `seq_len` matches the current token position $t+1$ (maximum sequence length is 32).*

## Tolerance

* **Relative Error Tolerance**: $< 5.0\%$
* **Absolute Error Tolerance**: $< 0.05$
* **Measured relative error vs CPU reference**: `0.02107` (2.1%)
* **Measured absolute error vs CPU reference**: `0.00183` (0.18%)

> [!NOTE]
> The tolerance of $5.0\%$ is allowed because of the custom fast exponential polynomial approximation on the NPU, which is highly optimized for hardware execution speed and provides sufficient precision for attention probabilities.
