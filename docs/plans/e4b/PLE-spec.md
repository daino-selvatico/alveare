# Spec: Gemma-4-E4B Per-Layer Embedding (PLE) & Architecture

This document defines the exact mathematical specification for Gemma-4-E4B Per-Layer Embedding (PLE), sliding/global attention patterns, KV head splits, and global scaling factors as implemented in `llama.cpp` (`src/models/gemma4-iswa.cpp` and `src/llama-model.cpp`).

---

## 1. Per-Layer Inputs Computation per Token

Per-layer inputs are computed once per token at the start of the model forward pass, before iterating through the transformer decoder blocks.

### Code Anchors
- Tensor creation & loading: `src/llama-model.cpp#L4623-L4627`
- Lookup function `build_inp_per_layer()`: `src/models/gemma4-iswa.cpp#L264-L295`
- Projection function `project_per_layer_inputs()`: `src/models/gemma4-iswa.cpp#L302-L322`

### Constants & Tensor Shapes
- `n_embd` = 2560 (main hidden dimension)
- `n_embd_per_layer` = 256 (PLE input dimension per layer)
- `n_layer` = 42
- `n_vocab` = 262144
- `per_layer_token_embd.weight`: shape `[262144, 10752]` (`[n_vocab, n_embd_per_layer * n_layer]`)
- `per_layer_model_proj.weight`: shape `[2560, 10752]` (`[n_embd, n_embd_per_layer * n_layer]`)
- `per_layer_proj_norm.weight`: shape `[256]` (`[n_embd_per_layer]`)

### Step-by-Step Mathematics

#### 1. Token Embedding Lookup (`inp_per_layer`)
For input token IDs `tokens` (shape `[n_tokens]`):
$$\text{emb\_lookup} = \text{Lookup}(\text{per\_layer\_token\_embd.weight}, \text{tokens}) \quad \in \mathbb{R}^{10752 \times n\_tokens}$$
Reshape to $\mathbb{R}^{256 \times 42 \times n\_tokens}$. Scale by $\sqrt{256} = 16.0$:
$$\text{inp\_per\_layer\_scaled} = \text{Reshape}_{256 \times 42 \times n\_tokens}(\text{emb\_lookup}) \times \sqrt{256}$$

#### 2. Projection of Main Token Embedding (`per_layer_proj`)
Let $\text{inpL}$ be the main token embedding lookup ($\text{token\_embd}[tokens]$ scaled by $\sqrt{n\_embd} = \sqrt{2560}$).
$$\text{proj\_raw} = \text{per\_layer\_model\_proj.weight}^T \times \text{inpL} \quad \in \mathbb{R}^{10752 \times n\_tokens}$$
Scale by $\frac{1}{\sqrt{n\_embd}} = \frac{1}{\sqrt{2560}}$:
$$\text{proj\_scaled} = \text{proj\_raw} \times \frac{1}{\sqrt{2560}}$$
Reshape to $\mathbb{R}^{256 \times 42 \times n\_tokens}$.

Apply **RMSNorm** (`LLM_NORM_RMS`) across the 256-dim embedding vector using `per_layer_proj_norm.weight` and $\epsilon = 10^{-6}$:
$$\text{per\_layer\_proj\_normed} = \text{RMSNorm}\left(\text{proj\_scaled}, \text{weight}=\text{per\_layer\_proj\_norm.weight}, \epsilon=10^{-6}\right)$$

Where $\text{RMSNorm}(x, \gamma, \epsilon) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$.

#### 3. Combination
Add the normalized projected representation and scaled lookup table embedding, then scale by $\frac{1}{\sqrt{2}}$:
$$\text{inp\_per\_layer} = \left( \text{per\_layer\_proj\_normed} + \text{inp\_per\_layer\_scaled} \right) \times \frac{1}{\sqrt{2.0}}$$

Permute dimensions to shape $\mathbb{R}^{256 \times n\_tokens \times 42}$ so that for layer $l \in [0, 41]$, the per-layer input vector per token $\text{inp\_per\_layer}[:, :, l]$ has shape $[256, n\_tokens]$.

---

## 2. Per-Layer Injection Math & Residual Placement

### Code Anchors
- Layer PLE tensor allocation: `src/llama-model.cpp#L4691-L4695`
- Forward execution graph per layer: `src/models/gemma4-iswa.cpp#L203-L224`

### Per-Block Tensors (for layer $l$)
- `blk.l.inp_gate.weight`: shape `[2560, 256]` (`[n_embd, n_embd_per_layer]`)
- `blk.l.proj.weight`: shape `[256, 2560]` (`[n_embd_per_layer, n_embd]`)
- `blk.l.post_norm.weight`: shape `[2560]` (`[n_embd]`)

### Exact Block Control Flow & Order of Operations

1. **State entering PLE block ($x_{\text{in}}$)**:
   The hidden state $x_{\text{in}}$ entering the PLE block is the state **immediately after the FFN residual connection**:
   $$x_{\text{in}} = x_{\text{attn\_out}} + \text{FFN\_Block}(x_{\text{attn\_out}})$$
   Where:
   - For standard layers: $\text{FFN\_Block}(x) = \text{ffn\_post\_norm}\left( \text{FFN}(\text{ffn\_norm}(x)) \right)$
   - For MoE layers: $\text{FFN\_Block}(x) = \text{ffn\_post\_norm\_1}(\text{MLP}) + \text{ffn\_post\_norm\_2}(\text{MoE})$

2. **Input Gating (`inp_gate`)**:
   $$g = \text{blk.l.inp\_gate.weight}^T \times x_{\text{in}} \quad \in \mathbb{R}^{256 \times n\_tokens}$$
   Apply standard GELU activation (`ggml_gelu`):
   $$a = \text{GELU}(g)$$

3. **Gating with Per-Layer Input**:
   Elementwise multiply activated gate with layer $l$'s per-layer input vector $h_l = \text{inp\_per\_layer}[:, :, l]$:
   $$z = a \odot h_l \quad \in \mathbb{R}^{256 \times n\_tokens}$$

4. **Layer Projection (`proj`)**:
   Project 256-dim gated vector back to hidden space (2560-dim):
   $$p = \text{blk.l.proj.weight}^T \times z \quad \in \mathbb{R}^{2560 \times n\_tokens}$$

5. **Post Norm (`post_norm`)**:
   Apply RMSNorm across 2560 dimension with scale weight `blk.l.post_norm.weight`:
   $$y = \text{RMSNorm}\left(p, \text{weight}=\text{blk.l.post\_norm.weight}, \epsilon=10^{-6}\right)$$

6. **Residual Connection**:
   Add to input $x_{\text{in}}$:
   $$x_{\text{after\_ple}} = x_{\text{in}} + y$$

7. **Layer Output Scaling (`out_scale`)**:
   $$x_{\text{out}} = x_{\text{after\_ple}} \odot \text{blk.l.layer\_output_scale.weight}$$
   ($x_{\text{out}}$ becomes the input $inpL$ to layer $l+1$).

### Injection Point Summary
The PLE contribution is added as a **third residual branch** in the layer:
$$\text{Layer\_Out} = \left( x_{\text{in\_layer}} + \text{Attn\_Branch} + \text{FFN\_Branch} + \text{PLE\_Branch} \right) \odot \text{layer\_output\_scale}$$

---

## 3. Gemma-4-E4B Geometry, Sliding/Global Pattern & KV-Head Split

### Summary Table

| Parameter | Value / Specification | Reference |
|---|---|---|
| Total Hidden Layers ($n\_layer$) | **42** | `gemma4.block_count = 42` |
| Hidden Dimension ($n\_embd$) | **2560** | `gemma4.embedding_length = 2560` |
| Intermediate Dimension ($n\_ff$) | **10240** | `gemma4.feed_forward_length = 10240` |
| Attention Query Heads ($n\_head$) | **8** (all layers) | `gemma4.attention.head_count = 8` |
| Key-Value Heads ($n\_head\_kv$) | **2** (all layers) | `gemma4.attention.head_count_kv = 2` |
| SWA Head Dimension ($d_{k,\text{swa}}$) | **256** | `gemma4.attention.key_length_swa = 256` |
| Global Head Dimension ($d_{k,\text{global}}$) | **512** | `gemma4.attention.key_length = 512` |
| Sliding Window Size | **512** | `gemma4.attention.sliding_window = 512` |
| SWA Layer Pattern | 5 SWA / 1 Global (period 6) | `gemma4.attention.sliding_window_pattern` |
| Shared KV Layers ($n\_kv\_shared$) | **18** | `gemma4.attention.shared_kv_layers = 18` |

### SWA vs Global Layer Pattern (Period of 6)
- **Sliding Window Attention (SWA)** (`is_swa = True`, 35 layers):
  - Layer indices (0-based): `0..4`, `6..10`, `12..16`, `18..22`, `24..28`, `30..34`, `36..40`
  - $Q$ shape: `[2560, 2048]` ($8 \times 256$)
  - $K$ shape: `[2560, 512]` ($2 \times 256$)
  - $V$ shape: `[2560, 512]` ($2 \times 256$)
  - $O$ shape: `[2048, 2560]` ($8 \times 256 \to 2560$)
  - RoPE base: `1e4` (`rope.freq_base_swa`)
  - Sliding window: 512

- **Global Full Attention** (`is_swa = False`, 7 layers):
  - Layer indices (0-based): `5`, `11`, `17`, `23`, `29`, `35`, `41`
  - $Q$ shape: `[2560, 4096]` ($8 \times 512$)
  - $K$ shape: `[2560, 1024]` ($2 \times 512$)
  - $V$ shape: `[2560, 1024]` ($2 \times 512$)
  - $O$ shape: `[4096, 2560]` ($8 \times 512 \to 2560$)
  - RoPE base: `1e6` (`rope.freq_base`)
  - Full attention (no sliding window mask)

### KV Cache Sharing Rule (`n_kv_shared_layers = 18`)
- Layers `0` to `23` (`has_kv(l) == true`): Compute $K$ and $V$ projections and write into KV cache.
- Layers `24` to `41` (`has_kv(l) == false`): **Do not write new KV states**; reuse existing KV cache from earlier layers via ISWA graph inputs (`build_attn_inp_kv_iswa`).

---

## 4. Global Scale Factors & Softcapping

| Factor | Value | Equation / Formula | Source Code Location |
|---|---|---|---|
| Main Embedding Scale Factor | $\sqrt{2560} \approx 50.59644$ | $\text{inpL} = \text{Lookup}(W_{emb}, \text{tok}) \times \sqrt{2560}$ | `gemma4-iswa.cpp#L20` |
| PLE Lookup Scale Factor | $\sqrt{256} = 16.0$ | $\text{inp\_per\_layer} = \text{Lookup}(W_{ple}, \text{tok}) \times 16.0$ | `gemma4-iswa.cpp#L268` |
| PLE Model Proj Scale Factor | $\frac{1}{\sqrt{2560}} \approx 0.019764$ | $\text{proj} = (W_{\text{model\_proj}}^T \times \text{inpL}) \times \frac{1}{\sqrt{2560}}$ | `gemma4-iswa.cpp#L303` |
| PLE Input Blend Scale Factor | $\frac{1}{\sqrt{2.0}} \approx 0.70710678$ | $(\text{proj\_normed} + \text{ple\_scaled}) \times \frac{1}{\sqrt{2.0}}$ | `gemma4-iswa.cpp#L304` |
| Attention Scale ($f_{\text{attention\_scale}}$) | **1.0f** | No pre-attention scaling (Gemma4 standard) | `llama-model.cpp#L1617` |
| Final Logit Softcapping | **30.0f** | $\text{logits} = 30.0 \times \tanh\left( \frac{\text{raw\_logits}}{30.0} \right)$ | `gemma4-iswa.cpp#L250-L254` |
| RMSNorm Epsilon ($\epsilon$) | **1e-6** | $\text{RMSNorm}(x, \gamma, 10^{-6})$ | `gemma4-iswa.cpp` / GGUF metadata |
