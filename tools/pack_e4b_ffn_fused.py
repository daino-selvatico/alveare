import os
import sys
import numpy as np

def pack_ffn_fused_weights(w_gate, w_up, w_down, H, I, m_I, k_tile):
    m_H = k_tile

    if I % (32 * m_I) == 0:
        n_cores = 32
    elif I % (16 * m_I) == 0:
        n_cores = 16
    elif I % (8 * m_I) == 0:
        n_cores = 8
    elif I % (4 * m_I) == 0:
        n_cores = 4
    elif I % (2 * m_I) == 0:
        n_cores = 2
    else:
        n_cores = 1

    I_div_n_cores = I // n_cores
    num_blocks_I = I_div_n_cores // m_I
    chunks_per_gate_up = k_tile // 32

    core_buffers = []

    for c in range(n_cores):
        start_I = c * I_div_n_cores
        end_I = (c + 1) * I_div_n_cores
        
        w_gate_slice = w_gate[start_I:end_I]
        w_up_slice = w_up[start_I:end_I]
        
        start_block = start_I // 32
        end_block = end_I // 32
        w_down_slice = w_down[:, start_block * 20 : end_block * 20]
        
        core_bytes = []

        n_down_tiles = H // m_H
        n_passes = next(p for p in range(1, n_down_tiles + 1)
                        if n_down_tiles % p == 0 and H // p <= 1024)
        down_tiles_per_pass = n_down_tiles // n_passes

        # Phase 1: gate + up tiles
        for b_I in range(num_blocks_I):
            row_start = b_I * m_I
            row_end = (b_I + 1) * m_I
            for h_blk in range(H // k_tile):
                col_start_bytes = h_blk * chunks_per_gate_up * 20
                col_end_bytes = (h_blk + 1) * chunks_per_gate_up * 20
                core_bytes.append(w_gate_slice[row_start:row_end, col_start_bytes:col_end_bytes].tobytes())
                core_bytes.append(w_up_slice[row_start:row_end, col_start_bytes:col_end_bytes].tobytes())

        # Phase 2: down tiles
        for p in range(n_passes):
            for b_I in range(num_blocks_I):
                col_start_bytes = b_I * (m_I // 32) * 20
                col_end_bytes = (b_I + 1) * (m_I // 32) * 20
                for h_blk_down in range(p * down_tiles_per_pass, (p + 1) * down_tiles_per_pass):
                    row_start_down = h_blk_down * m_H
                    row_end_down = (h_blk_down + 1) * m_H
                    tile = w_down_slice[row_start_down:row_end_down, col_start_bytes:col_end_bytes]
                    core_bytes.append(tile.tobytes())

        core_buf = np.frombuffer(b"".join(core_bytes), dtype=np.uint8)
        core_buffers.append(core_buf)

    if n_cores in (16, 32):
        tile_size = m_I * (k_tile // 32) * 20
        n_cols = 8
        rows_per_col = n_cores // n_cols
        n_tiles = core_buffers[0].size // tile_size
        parts = []
        for col in range(n_cols):
            for t in range(n_tiles):
                off = t * tile_size
                for r in range(rows_per_col):
                    s = core_buffers[rows_per_col * col + r]
                    parts.append(s[off:off + tile_size])
        return np.concatenate(parts)

    return np.stack(core_buffers)

def pack_e4b_fused(model_dir="quantized_weights_gemma4-e4b", num_layers=42):
    H = 2560
    I = 10240
    m_I = 32
    k_tile = 256
    
    print(f"Packing fused FFN weights for {model_dir} (H={H}, I={I}, layers={num_layers})...")
    
    for l in range(num_layers):
        gate_path = os.path.join(model_dir, f"blk.{l}.ffn_gate.weight_packed.npy")
        up_path = os.path.join(model_dir, f"blk.{l}.ffn_up.weight_packed.npy")
        down_path = os.path.join(model_dir, f"blk.{l}.ffn_down.weight_packed.npy")
        out_path = os.path.join(model_dir, f"blk.{l}.ffn_fused.weight_packed.npy")
        
        if os.path.exists(out_path):
            print(f"Layer {l}: {out_path} already exists, skipping.")
            continue
            
        if not (os.path.exists(gate_path) and os.path.exists(up_path) and os.path.exists(down_path)):
            print(f"Layer {l}: missing gate/up/down files, skipping.")
            continue
            
        w_gate = np.load(gate_path)
        w_up = np.load(up_path)
        w_down = np.load(down_path)
        
        fused = pack_ffn_fused_weights(w_gate, w_up, w_down, H, I, m_I, k_tile)
        np.save(out_path, fused)
        print(f"Layer {l} packed: {out_path} ({fused.nbytes / (1024*1024):.2f} MB)")
        
    print("Done!")

if __name__ == "__main__":
    pack_e4b_fused()
