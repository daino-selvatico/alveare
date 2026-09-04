# 🐝 Alveare 3.0 Roadmap — Unified Multi-Device Engine & Real-Time Live Assistant

This roadmap defines the technical architecture, execution plan, verification gates, and benchmarks for **Alveare 3.0**.
Every task strictly follows the development cycle: **Implement -> Test -> Benchmark -> Optimize -> Commit & Push**.

---

## 🎯 Release 3.0 Goals & Definition of Done

1. **Tri-Hardware Execution (CPU, NPU, GPU)**:
   - Every supported model family (`gemma4`, `gemma4-e4b`, `gemma3`, `llama`, `whisper-base`, `whisper-large-v3-turbo`, `audio8-0.1b`, `audio8-0.6b`) must execute reliably across **CPU**, **NPU (AMD XDNA2)**, and **GPU (Radeon 890M / Vulkan / RDNA 3.5)**.
   - Dedicated, highly-tuned compute kernels for each target hardware.
   - GPU performance target: equal or surpass `llama.cpp` and `ollama` on the AMD Radeon 890M iGPU.

2. **Simultaneous Multi-Device Serving**:
   - Serve independent models across heterogeneous hardware at the same time without context switching or thrashing:
     - Example: **LLM Chat on GPU** + **STT (Whisper) on NPU** + **TTS (Audio8) on CPU**.
   - Granular device routing per model slot via CLI, REST API, and Web UI.

3. **Full-Duplex "Live" Mode (Voice-to-Voice Real-Time)**:
   - End-to-end continuous listening, thinking, and speaking pipeline.
   - Streaming VAD -> Low-latency STT -> Streaming Token Generation -> Sentence-chunked TTS streaming.
   - Interactive user barge-in (interruption handling).
   - Operable via WebSocket `/ws/live`, REST endpoints, CLI (`./alveare live`), and new dedicated **Live Studio** in Web UI.

4. **Web UI & CLI Modernization**:
   - New dedicated **Live View** tab in Web UI with real-time waveform visualizer, device hardware indicators, and audio streaming.
   - Multi-device status and model orchestration panel in Web UI.
   - CLI flags `--device [npu|cpu|gpu]` across all commands + new `alveare live` launcher.

5. **Complete Documentation**:
   - Comprehensive guides for multi-device setups, GPU compute engine, Live voice mode, and API references.

---

## 📋 Milestones & Task Breakdown

### Phase 1: Architecture & Multi-Device Abstraction Foundation
- [x] **M1.1**: Create `ROADMAP_3.0.md`, establish git branch `rc-3.0`, and open tracking PR. (Completed: PR #33 opened)
- [x] **M1.2**: Design unified C++ compute engine abstraction (`ComputeDevice` / `DeviceBackend`) in `runtime/cpp` decoupling model logic from raw XRT NPU calls. (Completed: `include/alveare/device.h`)
- [x] **M1.3**: Implement native high-performance **CPU Backend** in C++ (`CpuBackend`) using AVX2/AVX-512 + OpenMP for vectorized Q4_0 GEMV, batched GEMM, and fused FFN. (Completed: `include/alveare/cpu_backend.h`, `src/cpu_backend.cpp`)
- [x] **M1.4**: Unit tests and benchmarks for CPU Backend (verifying correctness against reference and measuring tok/s). (Completed: `cpu_backend_test` passed, `layer_test` passed, `gemma3` generated ~12 tok/s on CPU)

### Phase 2: High-Performance GPU Engine (Vulkan / RDNA 3.5 Compute)
- [x] **M2.1**: Implement **GPU Backend** (`GpuBackend`) in `runtime/cpp` leveraging Vulkan 1.4 compute pipelines with fast subgroup operations and cooperative matrix / tiled shared memory. (Completed: `include/alveare/gpu_backend.h`, `src/gpu_backend.cpp`)
- [x] **M2.2**: Integrate Vulkan shader kernels for Q4_0 dequantization, GEMV dot products, attention, and fused GeGLU/SwiGLU. (Completed: `kernels/gpu/gemv_q4_0.comp`, `kernels/gpu/gemm_q4_0.comp`, `spv_shaders.h`)
- [x] **M2.3**: Comprehensive GPU benchmark vs `llama.cpp` on Radeon 890M to meet/exceed throughput targets. (Completed: `gpu_backend_test` passed with exact reference parity; `gemma3` achieved **0.32s prefill** and **32-45+ tok/s decode** on AMD Radeon 890M)

### Phase 3: Speech & Audio Engines on Multi-Hardware (STT & TTS)
- [ ] **M3.1**: Enable Whisper STT across all devices: CPU (multi-threaded AVX2), NPU (XDNA2 C-API), GPU (Vulkan/ROCm/Torch).
- [ ] **M3.2**: Enable Audio8 TTS across all devices: CPU (OpenMP vector), NPU (XDNA2), GPU (Vulkan/ROCm/Torch).
- [ ] **M3.3**: Benchmarks for STT Real-Time Factor (RTF) and TTS Time-To-First-Audio (TTFA) across hardware combinations.

### Phase 4: Multi-Model Concurrent Orchestration & Server Refactor
- [ ] **M4.1**: Refactor `control_server.py` to support multi-model concurrent execution (independent slots for LLM, STT, and TTS with dedicated hardware devices).
- [ ] **M4.2**: Update OpenAI-compatible API to allow simultaneous chat, transcription, and speech synthesis without unloading models.
- [ ] **M4.3**: Integration tests for concurrent multi-model serving under load.

### Phase 5: Real-Time Full-Duplex "Live" Engine
- [ ] **M5.1**: Implement `LiveOrchestrator` with low-latency streaming pipeline: Audio In -> VAD -> STT -> LLM Stream -> Sentence Chunking -> TTS Stream -> Audio Out.
- [ ] **M5.2**: Implement user barge-in detection and instant cancellation of ongoing generation.
- [ ] **M5.3**: Create WebSocket `/ws/live` protocol for duplex binary audio + JSON event exchange.
- [ ] **M5.4**: Add CLI command `./alveare live` for interactive terminal/audio live mode.

### Phase 6: Web UI Dashboard Modernization (Live View & Multi-HW Controls)
- [ ] **M6.1**: Add dedicated **"Live"** studio view in React frontend with interactive microphone streaming, audio visualizer, latency telemetry, and barge-in.
- [ ] **M6.2**: Add multi-device configuration UI (select LLM device, STT device, TTS device dynamically).
- [ ] **M6.3**: Frontend unit and integration tests (Vitest) for Live components.

### Phase 7: Verification, Documentation, and 3.0 Release Polish
- [ ] **M7.1**: Run full regression test suite (CPU, NPU, GPU, Live, API).
- [ ] **M7.2**: Complete end-to-end benchmark suite and generate formal reports in `benchmarks/reports/`.
- [ ] **M7.3**: Update all documentation: `README.md`, `docs/live.md`, `docs/multi-device.md`, `docs/gpu-engine.md`.
- [ ] **M7.4**: Final review and prepare PR merge.

---

*Last Updated: 2026-09-04 (Release Candidate 3.0 Init)*
