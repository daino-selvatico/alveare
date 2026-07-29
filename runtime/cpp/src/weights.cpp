#include "alveare/weights.h"
#include "alveare/npy.h"
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <cstdlib>
#include "alveare/bf16.h"

namespace alveare {

static std::vector<float> load_float_npy(const std::string& path) {
    NpyArray arr;
    try {
        arr = load_npy(path);
    } catch (const std::exception& e) {
        return {};
    }
    if (!arr.data) {
        std::cerr << "Warning: Failed to load " << path << "\n";
        return {};
    }
    size_t num_elements = 1;
    for (size_t d : arr.shape) num_elements *= d;
    
    std::vector<float> vec;
    bool is_f16 = (arr.dtype == "<f2" || arr.dtype == "float16" || arr.dtype == "|f2");
    bool is_bf16 = (arr.dtype == "<bfloat16" || arr.dtype == "bfloat16");
    bool two_byte = (num_elements > 0 && arr.data_size / num_elements == 2);
    if (is_f16 || is_bf16 || two_byte) {
        size_t expected_elements = arr.data_size / 2;
        vec.resize(expected_elements);
        const uint16_t* ptr = reinterpret_cast<const uint16_t*>(arr.data);
        // Default unknown 2-byte tensors to bf16; only true IEEE float16 uses the
        // half decoder (mixing the two silently corrupts every value).
        for (size_t i = 0; i < expected_elements; ++i) {
            if (is_f16) {
                vec[i] = half_to_float(ptr[i]);
            } else {
                alveare::bf16 b;
                b.v = ptr[i];
                vec[i] = b.to_float();
            }
        }
    } else {
        vec.resize(arr.data_size / sizeof(float));
        std::memcpy(vec.data(), arr.data, arr.data_size);
    }
    
    free_npy(arr);
    return vec;
}

static std::vector<uint8_t> load_uint8_npy(const std::string& path) {
    NpyArray arr;
    try {
        arr = load_npy(path);
    } catch (const std::exception& e) {
        return {};
    }
    if (!arr.data) {
        return {};
    }
    std::vector<uint8_t> vec(arr.data_size);
    std::memcpy(vec.data(), arr.data, arr.data_size);
    free_npy(arr);
    return vec;
}

static std::vector<uint16_t> load_uint16_npy(const std::string& path) {
    NpyArray arr;
    try {
        arr = load_npy(path);
    } catch (const std::exception& e) {
        return {};
    }
    if (!arr.data) {
        return {};
    }
    std::vector<uint16_t> vec(arr.data_size / sizeof(uint16_t));
    std::memcpy(vec.data(), arr.data, arr.data_size);
    free_npy(arr);
    return vec;
}

static std::vector<uint8_t> pack_ffn_fused_weights(
    const std::vector<uint8_t>& w_gate,
    const std::vector<uint8_t>& w_up,
    const std::vector<uint8_t>& w_down,
    int H, int I, int m_I, int k_tile) {

    int m_H = k_tile;
    int n_cores = 1;
    if (I % (32 * m_I) == 0) n_cores = 32;
    else if (I % (16 * m_I) == 0) n_cores = 16;
    else if (I % (8 * m_I) == 0) n_cores = 8;
    else if (I % (4 * m_I) == 0) n_cores = 4;
    else if (I % (2 * m_I) == 0) n_cores = 2;

    int I_div_n_cores = I / n_cores;
    int num_blocks_I = I_div_n_cores / m_I;
    int chunks_per_gate_up = k_tile / 32;

    int gate_up_stride = (H / 32) * 20;
    int down_stride = (I / 32) * 20;

    // Build each core's tile stream separately so we can lay them out either
    // sequentially (<=8 cores) or interleaved per column (16-core memtile path).
    const int tile_size = m_I * (k_tile / 32) * 20;
    std::vector<std::vector<uint8_t>> per_core(n_cores);
    for (auto& v : per_core) v.reserve(size_t((w_gate.size() + w_up.size() + w_down.size()) / n_cores + tile_size));

    const int n_passes = 4;
    int down_tiles_per_pass = (H / m_H) / n_passes;

    for (int c = 0; c < n_cores; ++c) {
        std::vector<uint8_t>& stream = per_core[c];
        int start_I = c * I_div_n_cores;
        auto append_tile = [&](const std::vector<uint8_t>& w, int r_start, int r_end,
                               int c_start_bytes, int c_end_bytes, int stride) {
            for (int r = r_start; r < r_end; ++r) {
                const uint8_t* ptr = w.data() + r * stride + c_start_bytes;
                stream.insert(stream.end(), ptr, ptr + (c_end_bytes - c_start_bytes));
            }
        };

        // Phase 1: gate + up tiles for every I-block (interleaved, streamed once).
        for (int b_I = 0; b_I < num_blocks_I; ++b_I) {
            int row_start = start_I + b_I * m_I;
            int row_end = start_I + (b_I + 1) * m_I;
            for (int h_blk = 0; h_blk < H / k_tile; ++h_blk) {
                int col_start_bytes = h_blk * chunks_per_gate_up * 20;
                int col_end_bytes = (h_blk + 1) * chunks_per_gate_up * 20;
                append_tile(w_gate, row_start, row_end, col_start_bytes, col_end_bytes, gate_up_stride);
                append_tile(w_up, row_start, row_end, col_start_bytes, col_end_bytes, gate_up_stride);
            }
        }

        // Phase 2: down tiles, per H-output pass, per I-block.
        for (int p = 0; p < n_passes; ++p) {
            for (int b_I = 0; b_I < num_blocks_I; ++b_I) {
                int start_block = start_I / 32;
                int col_start_bytes_down = start_block * 20 + b_I * (m_I / 32) * 20;
                int col_end_bytes_down = start_block * 20 + (b_I + 1) * (m_I / 32) * 20;
                int pass_end = (p == n_passes - 1) ? (H / m_H) : (p + 1) * down_tiles_per_pass;
                for (int h_blk_down = p * down_tiles_per_pass;
                     h_blk_down < pass_end; ++h_blk_down) {
                    int row_start_down = h_blk_down * m_H;
                    int row_end_down = (h_blk_down + 1) * m_H;
                    append_tile(w_down, row_start_down, row_end_down, col_start_bytes_down, col_end_bytes_down, down_stride);
                }
            }
        }
    }

    std::vector<uint8_t> fused;
    fused.reserve(w_gate.size() + w_up.size() + w_down.size());

    if (n_cores == 16 || n_cores == 32) {
        // Interleave the rows_per_col cores of each column tile-by-tile so the
        // kernel's per-column weight fill is contiguous:
        //   col block = [c0.t0, c1.t0, ..., c0.t1, c1.t1, ...]
        // Matches the kernel's split([0, tile_size, ...]) into the column's rows.
        const int n_cols = 8;
        const int rows_per_col = n_cores / n_cols;  // 2 or 4
        size_t per_core_bytes = per_core[0].size();
        int n_tiles = static_cast<int>(per_core_bytes / tile_size);
        for (int col = 0; col < n_cols; ++col) {
            for (int t = 0; t < n_tiles; ++t) {
                size_t off = size_t(t) * tile_size;
                for (int r = 0; r < rows_per_col; ++r) {
                    const std::vector<uint8_t>& s = per_core[rows_per_col * col + r];
                    fused.insert(fused.end(), s.begin() + off, s.begin() + off + tile_size);
                }
            }
        }
    } else {
        for (int c = 0; c < n_cores; ++c)
            fused.insert(fused.end(), per_core[c].begin(), per_core[c].end());
    }

    return fused;
}

ModelWeights load_weights(const std::string& dir, const ModelConfig& config, NpuRegistry& reg) {
    ModelWeights mw;
    mw.token_embd = load_float_npy(dir + "/token_embd.npy");
    mw.output_norm = load_float_npy(dir + "/output_norm.weight.npy");
    mw.lm_head = load_uint8_npy(dir + "/lm_head_packed.npy");

    if (config.per_layer_input > 0) {
        mw.per_layer_token_embd_f16 = load_uint16_npy(dir + "/per_layer_token_embd.npy");
        mw.per_layer_model_proj_packed = load_uint8_npy(dir + "/per_layer_model_proj_packed.npy");
        mw.per_layer_proj_norm = load_float_npy(dir + "/per_layer_proj_norm.weight.npy");
    }

    // Upload the LM head to the NPU as row-tiles when a matching gemv kernel is
    // available (the packed head is huge -- vocab x hidden -- so a CPU matmul
    // dominates decode). It is stored on disk as (vocab, K/32*20) uint8 with K
    // already padded to the kernel's K; tile it into chunk_N-row gemv weights.
    if (!mw.lm_head.empty()) {
        int lm_K = config.get_padded_hidden_size();
        int row_bytes = (lm_K / 32) * 20;
        int vocab = static_cast<int>(mw.lm_head.size() / row_bytes);
        const int chunk_N = 16384; // MAX_N supported by the harvested gemv kernels
        if (row_bytes > 0 && reg.has_gemv(chunk_N, lm_K) && vocab % chunk_N == 0) {
            size_t chunk_bytes = static_cast<size_t>(chunk_N) * row_bytes;
            for (int c = 0; c < vocab / chunk_N; ++c) {
                const uint8_t* ptr = mw.lm_head.data() + static_cast<size_t>(c) * chunk_bytes;
                mw.lm_head_chunks.push_back(reg.create_gemv_weight(chunk_N, lm_K, ptr, chunk_bytes));
            }
            mw.lm_head_vocab = vocab;
            mw.lm_head_K = lm_K;
            mw.lm_head_chunk_N = chunk_N;
            mw.lm_head.clear();
            mw.lm_head.shrink_to_fit();
            std::cout << "LM head on NPU: " << mw.lm_head_chunks.size()
                      << " tiles of (" << chunk_N << ", " << lm_K << ")\n";
        } else {
            std::cout << "LM head: no NPU gemv kernel for (" << chunk_N << ", " << lm_K
                      << "), falling back to CPU dequant.\n";
        }
    }

    int K_attn = config.hidden_size;
    int N_q = config.num_attention_heads * config.head_dim;
    int N_kv = config.num_key_value_heads * config.head_dim;
    int N_out = config.hidden_size;

    for (int l = 0; l < config.num_hidden_layers; ++l) {
        std::cout << "Loading weights for layer " << l << " ...\r" << std::flush;
        LayerWeights lw;
        lw.attn_norm = load_float_npy(dir + "/blk." + std::to_string(l) + ".attn_norm.weight.npy");
        lw.ffn_norm = load_float_npy(dir + "/blk." + std::to_string(l) + ".ffn_norm.weight.npy");

        if (config.model_type == "gemma3" || config.is_gemma4()) {
            lw.post_attention_norm = load_float_npy(dir + "/blk." + std::to_string(l) + ".post_attention_norm.weight.npy");
            lw.post_ffw_norm = load_float_npy(dir + "/blk." + std::to_string(l) + ".post_ffw_norm.weight.npy");
            lw.q_norm = load_float_npy(dir + "/blk." + std::to_string(l) + ".attn_q_norm.weight.npy");
            lw.k_norm = load_float_npy(dir + "/blk." + std::to_string(l) + ".attn_k_norm.weight.npy");
        }

        // Gemma-4 applies a per-layer scalar to the block output.
        if (config.is_gemma4()) {
            auto os = load_float_npy(dir + "/blk." + std::to_string(l) + ".layer_output_scale.weight.npy");
            if (!os.empty()) lw.output_scale = os[0];
        }

        if (config.per_layer_input > 0) {
            lw.inp_gate_bytes = load_uint8_npy(dir + "/blk." + std::to_string(l) + ".inp_gate.weight_packed.npy");
            lw.proj_bytes = load_uint8_npy(dir + "/blk." + std::to_string(l) + ".proj.weight_packed.npy");
            lw.post_norm = load_float_npy(dir + "/blk." + std::to_string(l) + ".post_norm.weight.npy");
        }

        // QKV, O projections
        bool is_sliding = (config.is_gemma4() && (l + 1) % config.sliding_pattern_period != 0);
        int l_N_q = N_q;
        int l_N_kv = N_kv;
        int l_N_out = N_out;
        int K_attn_padded = K_attn;
        if (config.is_gemma4()) {
            int h_dim = is_sliding ? config.head_dim : config.head_dim_global;
            l_N_q = config.num_attention_heads * h_dim;
            if (config.model_type == "gemma4-e4b") {
                l_N_kv = config.num_key_value_heads * h_dim;
            } else {
                l_N_kv = 2048; // 12B GGUF pads K to 2048
            }
            l_N_out = config.get_padded_hidden_size();
            K_attn_padded = config.get_padded_hidden_size();
        }

        std::string act_type = (config.model_type == "gemma3" || config.is_gemma4()) ? "gelu" : "silu";

        std::string q_path = dir + "/blk." + std::to_string(l) + ".attn_q.weight_packed.npy";
        std::string k_path = dir + "/blk." + std::to_string(l) + ".attn_k.weight_packed.npy";
        std::string v_path = dir + "/blk." + std::to_string(l) + ".attn_v.weight_packed.npy";
        std::string o_path = dir + "/blk." + std::to_string(l) + ".attn_output.weight_packed.npy";

        int n_kv_start = config.num_hidden_layers - config.shared_kv_layers;
        bool has_kv = (config.shared_kv_layers == 0) || (l < n_kv_start);

        // gemma4: fuse Q/K/V into a single resident weight (concatenated along
        // the output dim) so the three projections run as ONE gemv — one NPU
        // launch and one kernel-shape context, avoiding ~2.6 ms of per-shape
        // context-switch overhead per extra call. The packed layout is row-major
        // (N, K/32*20), so concatenating along N is just concatenating bytes.
        if (config.is_gemma4()) {
            NpyArray q_arr = load_npy(q_path);
            lw.n_q = l_N_q;
            lw.n_kv = has_kv ? l_N_kv : 0;
            std::vector<uint8_t> qkv;
            const uint8_t* qd = static_cast<const uint8_t*>(q_arr.data);

            if (!has_kv) {
                // Layers 24-41 (shared KV) only have Q projection
                lw.n_qkv = l_N_q;
                lw.w_qkv = reg.create_gemv_weight(lw.n_qkv, K_attn_padded, q_arr.data, q_arr.data_size);
            } else if (is_sliding) {
                // q ++ k ++ v  (N_qkv = N_q + 2*N_kv)
                NpyArray k_arr = load_npy(k_path);
                NpyArray v_arr = load_npy(v_path);
                const uint8_t* kd = static_cast<const uint8_t*>(k_arr.data);
                const uint8_t* vd = static_cast<const uint8_t*>(v_arr.data);
                lw.n_qkv = l_N_q + 2 * l_N_kv;
                qkv.reserve(q_arr.data_size + k_arr.data_size + v_arr.data_size);
                qkv.insert(qkv.end(), qd, qd + q_arr.data_size);
                qkv.insert(qkv.end(), kd, kd + k_arr.data_size);
                qkv.insert(qkv.end(), vd, vd + v_arr.data_size);
                free_npy(k_arr);
                free_npy(v_arr);
                lw.w_qkv = reg.create_gemv_weight(lw.n_qkv, K_attn_padded, qkv.data(), qkv.size());
            } else {
                // q ++ k  (global layers reuse k for v; N_qkv = N_q + N_kv)
                NpyArray k_arr = load_npy(k_path);
                const uint8_t* kd = static_cast<const uint8_t*>(k_arr.data);
                lw.n_qkv = l_N_q + l_N_kv;
                qkv.reserve(q_arr.data_size + k_arr.data_size);
                qkv.insert(qkv.end(), qd, qd + q_arr.data_size);
                qkv.insert(qkv.end(), kd, kd + k_arr.data_size);
                free_npy(k_arr);
                lw.w_qkv = reg.create_gemv_weight(lw.n_qkv, K_attn_padded, qkv.data(), qkv.size());
            }
            free_npy(q_arr);
        } else {
            NpyArray q_arr = load_npy(q_path);
            lw.w_q = reg.create_gemv_weight(l_N_q, K_attn_padded, q_arr.data, q_arr.data_size);
            free_npy(q_arr);

            NpyArray k_arr = load_npy(k_path);
            lw.w_k = reg.create_gemv_weight(l_N_kv, K_attn_padded, k_arr.data, k_arr.data_size);
            free_npy(k_arr);

            NpyArray v_arr = load_npy(v_path);
            lw.w_v = reg.create_gemv_weight(l_N_kv, K_attn_padded, v_arr.data, v_arr.data_size);
            free_npy(v_arr);
        }

        NpyArray o_arr = load_npy(o_path);
        if (config.is_gemma4()) {
            int target_N = 4096;
            int row_bytes = o_arr.data_size / l_N_out;
            if (row_bytes > 0 && reg.has_gemv(target_N, l_N_q)) {
                lw.o_gemv_n = target_N;
                std::vector<uint8_t> o_pad(size_t(target_N) * row_bytes, 0);
                std::memcpy(o_pad.data(), o_arr.data, o_arr.data_size);
                lw.w_o = reg.create_gemv_weight(target_N, l_N_q, o_pad.data(), o_pad.size());
            } else {
                lw.o_gemv_n = l_N_out;
                lw.w_o = reg.create_gemv_weight(l_N_out, l_N_q, o_arr.data, o_arr.data_size);
            }
        } else {
            lw.o_gemv_n = K_attn_padded;
            lw.w_o = reg.create_gemv_weight(K_attn_padded, l_N_q, o_arr.data, o_arr.data_size);
        }
        free_npy(o_arr);

        // FFN Fused
        int H_padded = config.get_padded_hidden_size();
        int I_padded = config.get_padded_intermediate_size();
        
        std::string ffn_path = dir + "/blk." + std::to_string(l) + ".ffn_fused.weight_packed.npy";
        NpyArray ffn_arr{};
        try {
            ffn_arr = load_npy(ffn_path);
        } catch (...) {
            ffn_arr.data = nullptr;
        }
        
        if (ffn_arr.data) {
            lw.w_ffn_fused = reg.create_ffn_fused_weight(H_padded, I_padded, act_type, ffn_arr.data, ffn_arr.data_size);
            free_npy(ffn_arr);
        } else {
            // Fallback: load gate, up, down and pack in C++
            std::cout << "\nPre-packing FFN fused weights for layer " << l << " ...\r" << std::flush;
            std::string gate_path = dir + "/blk." + std::to_string(l) + ".ffn_gate.weight_packed.npy";
            std::string up_path = dir + "/blk." + std::to_string(l) + ".ffn_up.weight_packed.npy";
            std::string down_path = dir + "/blk." + std::to_string(l) + ".ffn_down.weight_packed.npy";

            auto w_gate = load_uint8_npy(gate_path);
            auto w_up = load_uint8_npy(up_path);
            auto w_down = load_uint8_npy(down_path);

            if (w_gate.empty() || w_up.empty() || w_down.empty()) {
                throw std::runtime_error("Missing FFN weights for layer " + std::to_string(l));
            }

            int k_tile = (config.hidden_size == 1152) ? 128 : 256;
            auto fused = pack_ffn_fused_weights(w_gate, w_up, w_down, H_padded, I_padded, 32, k_tile);

            // Register the fused FFN on the NPU iff a matching kernel exists in the
            // manifest; otherwise keep the raw gate/up/down bytes for the CPU
            // fallback. (The old `H_padded % 1024 == 0` gate wrongly excluded e4b's
            // H=2560, forcing a ~17s/token CPU FFN even though the 2560x10240 kernel
            // is built.)
            // TODO(e4b): the e4b FFN kernel (H=2560, non-power-of-2) currently HANGS
            // the NPU — force the CPU fallback for e4b until the kernel is fixed;
            // override with ALVEARE_E4B_NPU_FFN=1 to debug the kernel.
            bool e4b_ffn_blocked = (config.model_type == "gemma4-e4b" &&
                                    std::getenv("ALVEARE_E4B_NPU_FFN") == nullptr);
            if (reg.has_ffn_fused(H_padded, I_padded, act_type) && !e4b_ffn_blocked) {
                lw.w_ffn_fused = reg.create_ffn_fused_weight(H_padded, I_padded, act_type, fused.data(), fused.size());
            } else {
                lw.w_ffn_fused = kInvalidWeight;
            }

            lw.ffn_gate_bytes = std::move(w_gate);
            lw.ffn_up_bytes = std::move(w_up);
            lw.ffn_down_bytes = std::move(w_down);
        }

        mw.layers.push_back(lw);
    }
    std::cout << "\nLoaded all weights.\n";
    return mw;
}

} // namespace alveare
