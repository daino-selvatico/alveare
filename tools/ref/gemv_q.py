import numpy as np
from ml_dtypes import bfloat16

def dequantize_q4_0(w_q4: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """
    Dequantize Q4_0 packed weights to bfloat16.
    
    Args:
        w_q4: uint8 array of shape (N, K // 2) containing packed int4 weights.
        scales: bfloat16 or float32 array of shape (N, K // 32) containing the block scales.
        
    Returns:
        Dequantized weight matrix of shape (N, K) in bfloat16.
    """
    N, K_half = w_q4.shape
    K = K_half * 2
    
    # Extract lower 4 bits (q0) and upper 4 bits (q1)
    q0 = (w_q4 & 0x0F).astype(np.int8)
    q1 = ((w_q4 >> 4) & 0x0F).astype(np.int8)
    
    # Sign extension from 4-bit to 8-bit signed integer
    q0[q0 >= 8] -= 16
    q1[q1 >= 8] -= 16
    
    # Interleave to reconstruct shape (N, K)
    q = np.empty((N, K), dtype=np.int8)
    q[:, 0::2] = q0
    q[:, 1::2] = q1
    
    # Expand scales from (N, K // 32) to (N, K) by repeating each scale 32 times
    scales_expanded = np.repeat(scales.astype(np.float32), 32, axis=1)
    
    # Dequantize
    w_dequant = q.astype(np.float32) * scales_expanded
    
    return w_dequant.astype(bfloat16)

def dequantize_combined(w_combined: np.ndarray) -> np.ndarray:
    """
    Dequantize the combined (packed weights + scales + padding) layout to bfloat16.
    
    Args:
        w_combined: uint8 array of shape (N, K // 32 * 20)
        
    Returns:
        Dequantized weight matrix of shape (N, K) in bfloat16.
    """
    N, K_blocks_20 = w_combined.shape
    K_blocks = K_blocks_20 // 20
    
    # Reshape to (N, K_blocks, 20) to avoid loops
    w_reshaped = w_combined.reshape(N, K_blocks, 20)
    
    # Extract weights: first 16 bytes of each block
    w_q4 = w_reshaped[:, :, :16].copy().reshape(N, K_blocks * 16)
    
    # Extract scales: bytes 16 and 17 of each block
    scales_bytes = w_reshaped[:, :, 16].astype(np.uint16) | (w_reshaped[:, :, 17].astype(np.uint16) << 8)
    scales = scales_bytes.view(bfloat16)
    
    return dequantize_q4_0(w_q4, scales)

def gemv_q(w_q4: np.ndarray, scales: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Quantized matrix-vector multiply (GEMV) reference.
    Computes y = W @ x where W is block quantized.
    """
    W_dequant = dequantize_q4_0(w_q4, scales)
    y_fp32 = W_dequant.astype(np.float32) @ x.astype(np.float32)
    return y_fp32.astype(bfloat16)

def gemv_q_combined(w_combined: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Quantized matrix-vector multiply (GEMV) using combined weights.
    Computes y = W @ x.
    """
    W_dequant = dequantize_combined(w_combined)
    y_fp32 = W_dequant.astype(np.float32) @ x.astype(np.float32)
    return y_fp32.astype(bfloat16)
