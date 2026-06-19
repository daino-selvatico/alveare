import numpy as np
from ml_dtypes import bfloat16
from gguf import GGUFReader

def quantize_to_q4_0(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantize an FP32/FP16 matrix W of shape (N, K) to Q4_0 layout.
    
    Args:
        W: Float matrix of shape (N, K) to quantize. K must be a multiple of 32.
        
    Returns:
        w_q4: uint8 array of shape (N, K // 2) containing packed int4 weights.
        scales: bfloat16 array of shape (N, K // 32) containing the block scales.
    """
    N, K = W.shape
    assert K % 32 == 0, f"K dimension must be a multiple of 32, but got {K}"
    
    # Reshape W to group elements into blocks of 32 along K dimension
    # W_blocks shape: (N, K // 32, 32)
    W_blocks = W.reshape(N, K // 32, 32)
    
    # Calculate scale for each block: max(abs(block)) / 7.0
    max_vals = np.max(np.abs(W_blocks), axis=2)
    scales = max_vals / 7.0
    
    # Handle division by zero
    scales[scales == 0.0] = 1.0
    
    # Broadcast scales to match W_blocks shape
    scales_expanded = np.expand_dims(scales, axis=2)
    
    # Quantize: round(val / scale)
    q_blocks = np.round(W_blocks / scales_expanded).astype(np.int32)
    
    # Clip to signed 4-bit range: [-8, 7]
    q_blocks = np.clip(q_blocks, -8, 7)
    
    # Reshape back to (N, K)
    q = q_blocks.reshape(N, K)
    
    # Pack weights: two 4-bit values per uint8 byte
    # q0 (lower 4 bits) corresponds to even columns: W[:, 2*i]
    # q1 (upper 4 bits) corresponds to odd columns: W[:, 2*i+1]
    q0 = q[:, 0::2]
    q1 = q[:, 1::2]
    
    w_q4 = ((q0 & 0x0F) | ((q1 & 0x0F) << 4)).astype(np.uint8)
    
    # Return packed weights and scale cast to bfloat16
    return w_q4, scales.astype(bfloat16)

def pack_to_combined(w_q4: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """
    Pack w_q4 (N, K // 2) and scales (N, K // 32) into a single combined
    array of shape (N, K // 32 * 20) of uint8.
    
    Each block of 32 elements corresponds to:
    - 16 bytes of packed weights (w_q4)
    - 2 bytes of scale (scales)
    - 2 bytes of padding (0)
    """
    N, K_half = w_q4.shape
    K = K_half * 2
    K_blocks = K // 32
    
    # Create combined array
    w_combined = np.zeros((N, K_blocks * 20), dtype=np.uint8)
    
    # Cast scales to bfloat16 and view as uint16 to get raw bytes
    scales_bf16 = scales.astype(bfloat16)
    scales_bytes = scales_bf16.view(np.uint16)
    
    for b in range(K_blocks):
        # 16 bytes of weights
        w_combined[:, b*20 : b*20 + 16] = w_q4[:, b*16 : (b+1)*16]
        
        # 2 bytes of scale
        scale_uint16 = scales_bytes[:, b]
        w_combined[:, b*20 + 16] = scale_uint16 & 0xFF
        w_combined[:, b*20 + 17] = (scale_uint16 >> 8) & 0xFF
        
        # 2 bytes of padding (already initialized to 0)
        
    return w_combined

def load_gguf_tensor(file_path: str, tensor_name: str) -> np.ndarray:
    """
    Load a tensor from a local GGUF file.
    """
    reader = GGUFReader(file_path)
    for tensor in reader.tensors:
        if tensor.name == tensor_name:
            return tensor.data.astype(np.float32)
    raise ValueError(f"Tensor '{tensor_name}' not found in {file_path}")
