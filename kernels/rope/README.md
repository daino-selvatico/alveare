# RoPE Kernel

This kernel applies Rotary Position Embeddings (RoPE) using Llama's split-half rotation style on the AMD Ryzen AI NPU.

## Design

For each head of size $H$ in the input activation vector $x$, the split-half rotation is defined as:

$$y_i = x_i \cos(i) - x_{i + H/2} \sin(i) \quad \text{for } 0 \le i < H/2$$
$$y_{i + H/2} = x_{i + H/2} \cos(i) + x_i \sin(i) \quad \text{for } 0 \le i < H/2$$

### Implementation Details
* **2-Input DMA Limit Resolution**: Ryzen AI AIE tiles support a maximum of 2 input DMA channels. To pass the activation vector and the embedding coefficients without exhausting channels, the `cos` and `sin` tables are concatenated in host memory into a single input buffer `cos_sin` of shape `(2 * H,)`.
* **Precision**: Operations are computed in `float32` to minimize accumulation and transformation errors before casting back to `bfloat16`.

## Host ABI

```c
void rope(
    const bfloat16 *restrict x,
    const bfloat16 *restrict cos_sin,
    bfloat16 *restrict y
);
```

## Shapes and Data Types

| Argument | Description | Shape | Data Type |
| :--- | :--- | :--- | :--- |
| `x` | Input query or key activation vector | `(K,)` | `bfloat16` |
| `cos_sin` | Packed cos and sin positional embedding factors | `(2 * H,)` | `bfloat16` |
| `y` | Output rotated vector | `(K,)` | `bfloat16` |

*Default `H = 64` (head dimension). `K = 2048` for Query, `K = 512` for Key.*

## Tolerance

* **Relative Error Tolerance**: $< 1.0\%$
* **Absolute Error Tolerance**: $< 0.02$
* **Measured relative error vs CPU reference**: `0.00441` (0.44%)
