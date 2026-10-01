# ⚡ Alveare 3.0: Tri-Hardware Heterogeneous Concurrent Execution

Alveare 3.0 breaks the single-device paradigm by introducing **Tri-Hardware Concurrent Execution**. A single AMD Ryzen AI Mini PC can now simultaneously run three independent AI models across its three distinct silicon engines:

```mermaid
graph TD
    User([User Voice / Input]) --> Mic[Microphone / Audio Input]
    Mic -->|Raw PCM 16kHz| STT[Whisper STT Slot]
    
    subgraph "Tri-Hardware Routing"
        STT -->|Transcribed Text| NPU["⚡ AMD Ryzen AI NPU (XDNA2)<br/>Whisper STT (<600ms latency)"]
        NPU --> LLM[LLM Reasoning Slot]
        LLM -->|Prompt Tokens| GPU["🎮 AMD Radeon 890M GPU (Vulkan 1.4)<br/>Gemma-3-1B (35-45 tok/s)"]
        GPU -->|Streaming Sentences| TTS[Neural TTS Slot]
        TTS -->|Speech Synthesis| CPU["💻 AMD Ryzen Zen 5 CPU (AVX-512)<br/>Kokoro-82M (RTF ~0.10x, ~300ms)"]
    end
    
    CPU --> Speaker[Speaker / Web Audio Output]
```

---

## 🎯 The Tri-Hardware Optimal Partition

Running multiple models concurrently on a compact unified-memory PC requires careful hardware partitioning to avoid resource contention and thermal throttling:

| Engine Slot | Target Hardware | Why This Hardware? | Typical Latency / Throughput | Memory Footprint |
|---|---|---|---|---|
| **LLM Chat** | **AMD Radeon 890M GPU** | Maximum memory bandwidth and subgroup parallelism allow 35–45+ tok/s token streaming without CPU overhead. | 35–45+ tok/s | ~1.5–3.5 GB VRAM |
| **Whisper STT** | **AMD Ryzen AI NPU** | Continuous audio feature extraction and listening runs in background without touching GPU cores or stealing VRAM. | <600 ms STT | Dedicated SRAM + ~200 MB DRAM |
| **Kokoro TTS (Default)** | **AMD Ryzen CPU** | Non-autoregressive StyleTTS2 on 24 Zen 5 threads generates speech in single pass with RTF ~0.10x and zero GPU VRAM. | 250–450 ms TTFA | ~320 MB Host RAM |
| **Audio8 TTS (Optional)** | **AMD Ryzen NPU/CPU** | DualAR 10-codebook neural synthesis for zero-shot voice cloning from custom audio reference clips. | ~6–8s / chunk | ~340 MB–1.2 GB |

---

## 🏗️ Architecture: Independent Model Slots

In `runtime/py/control_server.py`, the backend maintains three isolated execution slots:

- `slots["llm"]`: Manages the C++ inference engine process (`alveare_runtime`) on GPU, NPU, or CPU.
- `slots["stt"]`: Manages the background Whisper worker on NPU, GPU, or CPU.
- `slots["tts"]`: Manages the background Audio8 neural TTS worker on CPU, GPU, or NPU.

Each slot has its own independent lifecycle:
- Starting or stopping one slot does **not** interrupt or restart other running slots.
- Telemetry monitors each device independently (NPU utilization via XRT, GPU utilization & VRAM via Linux sysfs, CPU load via psutil).

---

## 🔌 REST API Endpoints for Slots

### Start a Specific Model Slot
`POST /api/control/slot/start`
```json
{
  "slot_type": "llm",
  "model": "gemma4-e4b",
  "device": "gpu"
}
```

### Stop a Specific Model Slot
`POST /api/control/slot/stop`
```json
{
  "slot_type": "stt"
}
```

### Start Coordinated Tri-Hardware Live Profile
`POST /api/live/start`
```json
{
  "llm_model": "gemma4-e4b",
  "llm_device": "gpu",
  "stt_model": "whisper-large-v3-turbo",
  "stt_device": "cpu",
  "tts_model": "hexgrad/Kokoro-82M",
  "tts_device": "cpu"
}
```

### Check Slot Status and Telemetry
`GET /api/status`
Returns status for each slot (`llm`, `stt`, `tts`), GPU usage (`percent`, `vram_used_mb`, `vram_total_mb`), and NPU usage.

---

## 💻 CLI Usage

```bash
# Serve LLM on GPU
./alveare serve gemma4-e4b --device gpu

# Serve LLM on NPU
./alveare serve gemma4-e4b --device npu

# Serve LLM on CPU
./alveare serve gemma4-e4b --device cpu

# Launch the interactive full-duplex live session using all three engines
./alveare live
```
