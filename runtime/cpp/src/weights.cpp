#include "alveare/weights.h"
#include "alveare/npy.h"
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include "alveare/bf16.h"

namespace alveare {

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

    // N_PASSES MUST match ffn_fused.py exactly (smallest divisor of H//m_H with
    // H/N_PASSES <= 1024) — else the packed down-tile layout mismatches the kernel's
    // per-pass consumption and the NPU DMA hangs. 12B (H=4096) -> 4; e4b (H=2560) -> 5.
    const int n_down_tiles = H / m_H;
    int n_passes = 4;
    for (int p = 1; p <= n_down_tiles; ++p) {
        if (n_down_tiles % p == 0 && H / p <= 1024) { n_passes = p; break; }
    }
    int down_tiles_per_pass = n_down_tiles / n_passes;

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

ModelWeights load_weights(const std::string& dir, const ModelConfig& config, ComputeDevice& reg) {
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
        } else if (config.model_type == "gemma3") {
            l_N_q = 2048;
            l_N_kv = 2048;
            l_N_out = 2048;
            K_attn_padded = 2048;
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
        int qkv_K = 0;   // K of the fused-QKV kernel (for O context sharing below)
        // ALVEARE_ONESHAPE keeps the raw QKV/O bytes so they can be re-registered
        // zero-padded onto one shared kernel shape (see the oneshape block below).
        static const bool os_on = (std::getenv("ALVEARE_NO_ONESHAPE") == nullptr &&
                                   (std::getenv("ALVEARE_ONESHAPE") != nullptr || config.is_gemma4()));
        std::vector<uint8_t> os_qkv_src, os_o_src;
        int os_o_rows = 0;
        if (config.is_gemma4()) {
            NpyArray q_arr = load_npy(q_path);
            lw.n_q = l_N_q;
            lw.n_kv = has_kv ? l_N_kv : 0;
            std::vector<uint8_t> qkv;
            const uint8_t* qd = static_cast<const uint8_t*>(q_arr.data);
            int K_q = (l_N_q > 0 && q_arr.data_size % l_N_q == 0)
                    ? static_cast<int>((q_arr.data_size / l_N_q / 20) * 32)
                    : K_attn_padded;
            qkv_K = K_q;

            if (!has_kv) {
                // Layers 24-41 (shared KV) only have Q projection
                lw.n_qkv = l_N_q;
                if (reg.has_gemv(lw.n_qkv, K_q)) {
                    lw.w_qkv = reg.create_gemv_weight(lw.n_qkv, K_q, q_arr.data, q_arr.data_size);
                }
                os_qkv_src.assign(qd, qd + q_arr.data_size);
            } else if (is_sliding || config.model_type == "gemma4-e4b" || config.model_type == "e4b") {
                // q ++ k ++ v  (N_qkv = N_q + 2*N_kv). e4b has a real V on EVERY
                // layer (kv_heads=2 uniformly); only the 12B's global layers are
                // MQA and tie V=K (the `else` branch below).
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
                if (reg.has_gemv(lw.n_qkv, K_q)) {
                    lw.w_qkv = reg.create_gemv_weight(lw.n_qkv, K_q, qkv.data(), qkv.size());
                }
                os_qkv_src = qkv;
            } else {
                // q ++ k  (global layers reuse k for v; N_qkv = N_q + N_kv)
                NpyArray k_arr = load_npy(k_path);
                const uint8_t* kd = static_cast<const uint8_t*>(k_arr.data);
                lw.n_qkv = l_N_q + l_N_kv;
                qkv.reserve(q_arr.data_size + k_arr.data_size);
                qkv.insert(qkv.end(), qd, qd + q_arr.data_size);
                qkv.insert(qkv.end(), kd, kd + k_arr.data_size);
                free_npy(k_arr);
                if (reg.has_gemv(lw.n_qkv, K_q)) {
                    lw.w_qkv = reg.create_gemv_weight(lw.n_qkv, K_q, qkv.data(), qkv.size());
                }
                os_qkv_src = qkv;
            }
            free_npy(q_arr);
        } else {
            NpyArray q_arr = load_npy(q_path);
            if (reg.has_gemv(l_N_q, K_attn_padded)) {
                lw.w_q = reg.create_gemv_weight(l_N_q, K_attn_padded, q_arr.data, q_arr.data_size);
            }
            free_npy(q_arr);

            NpyArray k_arr = load_npy(k_path);
            if (reg.has_gemv(l_N_kv, K_attn_padded)) {
                lw.w_k = reg.create_gemv_weight(l_N_kv, K_attn_padded, k_arr.data, k_arr.data_size);
            }
            free_npy(k_arr);

            NpyArray v_arr = load_npy(v_path);
            if (reg.has_gemv(l_N_kv, K_attn_padded)) {
                lw.w_v = reg.create_gemv_weight(l_N_kv, K_attn_padded, v_arr.data, v_arr.data_size);
            }
            free_npy(v_arr);
        }

        NpyArray o_arr = load_npy(o_path);
        if (config.is_gemma4()) {
            int target_N = 4096;
            int row_bytes = o_arr.data_size / l_N_out;
            int K_o = (row_bytes / 20) * 32;   // O's real input dim (= N_q)
            os_o_src.assign(static_cast<const uint8_t*>(o_arr.data),
                            static_cast<const uint8_t*>(o_arr.data) + o_arr.data_size);
            os_o_rows = l_N_out;

            // BEST CASE: zero-pad O in BOTH dims to the fused-QKV kernel's (n_qkv, K_q)
            // so O runs in the SAME kernel context as QKV — removing one ~2.6 ms
            // context switch per layer. Padded weight rows/cols are zero (a zero Q4_0
            // block has scale 0 → contributes nothing), and only the first
            // hidden_size outputs are read back. On e4b's sliding layers this turns
            // O from (2560,2048) into QKV's (3072,2560).
            if (row_bytes > 0 && lw.n_qkv >= l_N_out && qkv_K > K_o &&
                reg.has_gemv(lw.n_qkv, qkv_K)) {
                const int new_row_bytes = (qkv_K / 32) * 20;
                std::vector<uint8_t> o_pad(size_t(lw.n_qkv) * new_row_bytes, 0);
                const uint8_t* src = static_cast<const uint8_t*>(o_arr.data);
                for (int r = 0; r < l_N_out; ++r)
                    std::memcpy(o_pad.data() + size_t(r) * new_row_bytes,
                                src + size_t(r) * row_bytes, row_bytes);
                lw.o_gemv_n = lw.n_qkv;
                lw.o_gemv_k = qkv_K;
                lw.w_o = reg.create_gemv_weight(lw.n_qkv, qkv_K, o_pad.data(), o_pad.size());
            } else if (row_bytes > 0 && reg.has_gemv(target_N, l_N_q)) {
                lw.o_gemv_n = target_N;
                std::vector<uint8_t> o_pad(size_t(target_N) * row_bytes, 0);
                std::memcpy(o_pad.data(), o_arr.data, o_arr.data_size);
                lw.w_o = reg.create_gemv_weight(target_N, l_N_q, o_pad.data(), o_pad.size());
            } else if (reg.has_gemv(l_N_out, l_N_q)) {
                lw.o_gemv_n = l_N_out;
                lw.w_o = reg.create_gemv_weight(l_N_out, l_N_q, o_arr.data, o_arr.data_size);
            }
        } else if (reg.has_gemv(K_attn_padded, l_N_q)) {
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
        }

        std::string gate_path = dir + "/blk." + std::to_string(l) + ".ffn_gate.weight_packed.npy";
        std::string up_path = dir + "/blk." + std::to_string(l) + ".ffn_up.weight_packed.npy";
        std::string down_path = dir + "/blk." + std::to_string(l) + ".ffn_down.weight_packed.npy";

        std::vector<uint8_t> w_gate, w_up, w_down;
        try {
            w_gate = load_uint8_npy(gate_path);
            w_up = load_uint8_npy(up_path);
            w_down = load_uint8_npy(down_path);
        } catch (...) {}

        // ALVEARE_ONESHAPE: register the FFN as gemv tiles that reuse
        // the shared shape (TN, TK), so the whole layer runs in ONE hw context.
        int TN_pick = 0;
        if (os_on && qkv_K > 0) {
            const int max_tn = (5 * H_padded) / 4;   // 1.25x: keep tiles near hidden (3072 for hidden=2560)
            for (int cand : {3072, 2560, 4096, 5120, 6144, 8192}) {
                if (cand > max_tn) break;
                if (cand >= config.hidden_size && reg.has_gemv(cand, qkv_K)) {
                    TN_pick = cand; break;
                }
            }
        }
        if (TN_pick > 0 && !w_gate.empty() && !w_up.empty() && !w_down.empty()) {
            const int TN = TN_pick, TK = qkv_K;
            const size_t tile_row_bytes = size_t(TK / 32) * 20;
            const size_t gu_row_bytes = tile_row_bytes;
            const int I_rows = (gu_row_bytes && w_gate.size() % gu_row_bytes == 0)
                             ? int(w_gate.size() / gu_row_bytes) : 0;
            const int gu_rows = 2 * I_rows;                        // gate ++ up

            if (I_rows > 0 && I_rows % TK == 0) {
                for (int base = 0; base < gu_rows; base += TN) {
                    std::vector<uint8_t> tile(size_t(TN) * tile_row_bytes, 0);
                    for (int r = 0; r < TN && base + r < gu_rows; ++r) {
                        int gr = base + r;
                        const std::vector<uint8_t>& src = (gr < I_rows) ? w_gate : w_up;
                        size_t off = size_t(gr < I_rows ? gr : gr - I_rows) * gu_row_bytes;
                        if (off + gu_row_bytes <= src.size())
                            std::memcpy(tile.data() + size_t(r) * tile_row_bytes,
                                        src.data() + off, gu_row_bytes);
                    }
                    lw.os_gateup.push_back(reg.create_gemv_weight(TN, TK, tile.data(), tile.size()));
                }
                const size_t down_row_bytes = size_t(I_padded / 32) * 20;
                const int n_chunks = I_rows / TK;
                for (int c = 0; c < n_chunks; ++c) {
                    std::vector<uint8_t> tile(size_t(TN) * tile_row_bytes, 0);
                    for (int r = 0; r < config.hidden_size && r < TN; ++r) {
                        size_t off = size_t(r) * down_row_bytes + size_t(c) * tile_row_bytes;
                        if (off + tile_row_bytes <= w_down.size())
                            std::memcpy(tile.data() + size_t(r) * tile_row_bytes,
                                        w_down.data() + off, tile_row_bytes);
                    }
                    lw.os_down.push_back(reg.create_gemv_weight(TN, TK, tile.data(), tile.size()));
                }
                const size_t trb = size_t(TK / 32) * 20;
                if (!os_qkv_src.empty() && os_qkv_src.size() % trb == 0) {
                    const int rows = int(os_qkv_src.size() / trb);
                    for (int base = 0; base < rows; base += TN) {
                        std::vector<uint8_t> tile(size_t(TN) * trb, 0);
                        const int n = std::min(TN, rows - base);
                        std::memcpy(tile.data(), os_qkv_src.data() + size_t(base) * trb,
                                    size_t(n) * trb);
                        lw.os_qkv_tiles.push_back(
                            reg.create_gemv_weight(TN, TK, tile.data(), tile.size()));
                    }
                }
                if (!os_o_src.empty() && os_o_rows > 0) {
                    const size_t orb = os_o_src.size() / size_t(os_o_rows);
                    if (orb <= trb) {
                        for (int base = 0; base < os_o_rows; base += TN) {
                            std::vector<uint8_t> tile(size_t(TN) * trb, 0);
                            const int n = std::min(TN, os_o_rows - base);
                            for (int r = 0; r < n; ++r)
                                std::memcpy(tile.data() + size_t(r) * trb,
                                            os_o_src.data() + size_t(base + r) * orb, orb);
                            lw.os_o_tiles.push_back(
                                reg.create_gemv_weight(TN, TK, tile.data(), tile.size()));
                        }
                        lw.os_o_is_kchunked = false;
                    } else {
                        const int n_kchunks = static_cast<int>((orb + trb - 1) / trb);
                        for (int c = 0; c < n_kchunks; ++c) {
                            std::vector<uint8_t> tile(size_t(TN) * trb, 0);
                            const size_t c_offset = size_t(c) * trb;
                            const size_t c_bytes = (c_offset + trb <= orb) ? trb : (orb - c_offset);
                            for (int r = 0; r < os_o_rows && r < TN; ++r) {
                                std::memcpy(tile.data() + size_t(r) * trb,
                                            os_o_src.data() + size_t(r) * orb + c_offset, c_bytes);
                            }
                            lw.os_o_tiles.push_back(
                                reg.create_gemv_weight(TN, TK, tile.data(), tile.size()));
                        }
                        lw.os_o_is_kchunked = true;
                    }
                }
                lw.os_n = TN;
                lw.os_k = TK;
                if (l == 0)
                    std::cout << "[oneshape] FFN as " << lw.os_gateup.size() << "+"
                              << lw.os_down.size() << " gemv tiles of (" << TN << "," << TK << ")\n";
            }
        } else if (lw.w_ffn_fused == kInvalidWeight && !w_gate.empty() && !w_up.empty() && !w_down.empty()) {
            int k_tile = 256;
            auto fused = pack_ffn_fused_weights(w_gate, w_up, w_down, H_padded, I_padded, 32, k_tile);
            if (reg.has_ffn_fused(H_padded, I_padded, act_type)) {
                lw.w_ffn_fused = reg.create_ffn_fused_weight(H_padded, I_padded, act_type, fused.data(), fused.size());
            }
        }

        lw.ffn_gate_bytes = std::move(w_gate);
        lw.ffn_up_bytes = std::move(w_up);
        lw.ffn_down_bytes = std::move(w_down);

        mw.layers.push_back(lw);
    }

    // Load Medusa multi-head weights if present (medusa_head_0.npy, medusa_head_1.npy, ...)
    for (int h = 0; h < 4; ++h) {
        std::string medusa_path = dir + "/medusa_head_" + std::to_string(h) + ".npy";
        std::ifstream test_f(medusa_path);
        if (test_f.good()) {
            std::vector<float> head_w = load_float_npy(medusa_path);
            if (!head_w.empty()) {
                mw.medusa_heads.push_back(std::move(head_w));
            }
        }
    }
    mw.num_medusa_heads = static_cast<int>(mw.medusa_heads.size());
    if (mw.num_medusa_heads > 0) {
        std::cout << "[medusa] Loaded " << mw.num_medusa_heads << " speculative prediction heads.\n";
    }

    std::cout << "\nLoaded all weights.\n";
    return mw;
}

} // namespace alveare
