# RMSNorm Kernel

This kernel performs Root Mean Square Normalization (RMSNorm) on the AMD Ryzen AI NPU.

## Design

RMSNorm scales the input vector $x$ using a weight vector (gamma) $w$ and the reciprocal of the root mean square of the elements:

$$y_i = \frac{x_i}{\sqrt{\frac{1}{K} \sum_{j=1}^K x_j^2 + \epsilon}} \times w_i$$

### Implementation Details
* **Precision**: To preserve numerical stability and avoid precision loss across the model layer, accumulation and inverse square root calculations are performed in `float32`.
* **Hardware Intrinsics**: Uses `aie::invsqrt` for high-performance reciprocal square root calculation.
* **Epsilon**: Standard value of `1e-5` is hardcoded.

## Host ABI

```c
void rmsnorm(
    const bfloat16 *restrict x,
    const float *restrict w,
    bfloat16 *restrict y
);
```

## Shapes and Data Types

| Argument | Description | Shape | Data Type |
| :--- | :--- | :--- | :--- |
| `x` | Input activation vector | `(K,)` | `bfloat16` |
| `w` | Gamma scale weight vector | `(K,)` | `float32` |
| `y` | Output normalized vector | `(K,)` | `bfloat16` |

*Default `K = 2048`.*

## Tolerance

* **Relative Error Tolerance**: $< 1.0\%$
* **Absolute Error Tolerance**: $< 0.02$
* **Measured relative error vs CPU reference**: `0.00410` (0.41%)
