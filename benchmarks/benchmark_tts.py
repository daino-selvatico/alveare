"""
Comparative Benchmark Script for Alveare TTS on Audio8 0.1B and 0.6B
Tests device='cpu' and device='npu' (with NPULinear offload).
Measures:
- Latency (ms)
- Audio Duration (s)
- RTF (Real-Time Factor)
- Throughput (tokens/sec and frames/sec)
- Offloaded linear layers count
"""
import os
import sys
import time
import json
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from runtime.py.audio8_tts import Audio8TTS

TEST_PROMPTS = [
    "Ciao! Benvenuti su Alveare con accelerazione neurale AMD Ryzen AI.",
    "L'architettura neurale ad alta efficienza permette di generare voce naturale a bassissima latenza direttamente sul processore locale.",
]

def run_tts_benchmark():
    configurations = [
        {"model": "Audio8/Audio8-TTS-Preview-0.1b", "alias": "Audio8 0.1B", "device": "cpu"},
        {"model": "Audio8/Audio8-TTS-Preview-0.1b", "alias": "Audio8 0.1B", "device": "npu"},
        {"model": "Audio8/Audio8-TTS-Preview-0.6b", "alias": "Audio8 0.6B", "device": "cpu"},
        {"model": "Audio8/Audio8-TTS-Preview-0.6b", "alias": "Audio8 0.6B", "device": "npu"},
    ]

    results = []

    print("=" * 80)
    print("STARTING ALVEARE AUDIO8 TTS COMPARATIVE BENCHMARK")
    print("=" * 80)

    for cfg in configurations:
        model_id = cfg["model"]
        alias = cfg["alias"]
        device = cfg["device"]

        print(f"\n>>> Initializing {alias} on device='{device}'...")
        tts = Audio8TTS.get_instance(model_id=model_id, device=device)
        tts._ensure_loaded()
        print(f">>> {alias} [{device}] Worker initialized successfully.")

        # Warm-up run
        print(f">>> Warming up {alias} [{device}]...")
        warmup_res = tts.generate(text="Test di riscaldamento iniziale.", max_new_tokens=60)
        if warmup_res.get("status") != "success":
            print(f"WARMUP ERROR: {warmup_res}")

        # Benchmark runs
        prompt_results = []
        for i, text in enumerate(TEST_PROMPTS):
            print(f"  [Run {i+1}/{len(TEST_PROMPTS)}] Prompt: \"{text[:45]}...\"")
            t_start = time.perf_counter()
            res = tts.generate(text=text, max_new_tokens=300, temperature=0.8, do_sample=True)
            wall_time = (time.perf_counter() - t_start) * 1000.0

            if res.get("status") == "success":
                lat = res.get("latency_ms", wall_time)
                dur = res.get("duration_sec", 0.0)
                rtf = res.get("rtf", (lat / 1000.0) / dur if dur > 0 else 0.0)
                tok_s = res.get("tokens_per_sec", 0.0)
                frames_s = res.get("frames_per_sec", 0.0)
                num_tok = res.get("num_tokens", 0)
                num_fr = res.get("num_frames", 0)
                prompt_results.append({
                    "prompt": text,
                    "latency_ms": lat,
                    "duration_sec": dur,
                    "rtf": rtf,
                    "tokens_per_sec": tok_s,
                    "frames_per_sec": frames_s,
                    "num_tokens": num_tok,
                    "num_frames": num_fr,
                    "audio_path": res.get("audio_path")
                })
                print(f"    -> Latency: {lat:.1f} ms | Duration: {dur:.2f} s | RTF: {rtf:.3f} | Throughput: {tok_s:.1f} tok/s ({frames_s:.1f} frames/s)")
            else:
                print(f"    -> Error: {res.get('error')}")

        # Compute averages
        avg_lat = sum(r["latency_ms"] for r in prompt_results) / len(prompt_results) if prompt_results else 0
        avg_dur = sum(r["duration_sec"] for r in prompt_results) / len(prompt_results) if prompt_results else 0
        avg_rtf = sum(r["rtf"] for r in prompt_results) / len(prompt_results) if prompt_results else 0
        avg_tok_s = sum(r["tokens_per_sec"] for r in prompt_results) / len(prompt_results) if prompt_results else 0
        avg_frames_s = sum(r["frames_per_sec"] for r in prompt_results) / len(prompt_results) if prompt_results else 0

        results.append({
            "model": alias,
            "model_id": model_id,
            "device": device,
            "avg_latency_ms": round(avg_lat, 1),
            "avg_duration_sec": round(avg_dur, 2),
            "avg_rtf": round(avg_rtf, 3),
            "avg_tokens_per_sec": round(avg_tok_s, 1),
            "avg_frames_per_sec": round(avg_frames_s, 1),
            "runs": prompt_results
        })

    # Save benchmark results to JSON
    out_file = ROOT_DIR / "benchmarks" / "reports" / "audio8_tts_benchmark_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY RESULTS TABLE")
    print("=" * 80)
    print(f"| {'Modello':<12} | {'Device':<14} | {'Latenza Media (ms)':<18} | {'Durata Audio (s)':<16} | {'RTF (Real-Time)':<16} | {'Throughput (tok/s)':<18} | {'Speedup vs CPU':<14} |")
    print(f"|{'-'*14}|{'-'*16}|{'-'*20}|{'-'*18}|{'-'*18}|{'-'*20}|{'-'*16}|")

    # Map CPU baselines for speedup
    cpu_lat = {}
    for r in results:
        if r["device"] == "cpu":
            cpu_lat[r["model"]] = r["avg_latency_ms"]

    for r in results:
        baseline = cpu_lat.get(r["model"], r["avg_latency_ms"])
        speedup = f"{baseline / r['avg_latency_ms']:.2f}x" if r["avg_latency_ms"] > 0 else "1.00x"
        dev_str = f"NPU (offload)" if r["device"] == "npu" else "CPU"
        print(f"| {r['model']:<12} | {dev_str:<14} | {r['avg_latency_ms']:<18.1f} | {r['avg_duration_sec']:<16.2f} | {r['avg_rtf']:<16.3f} | {r['avg_tokens_per_sec']:<18.1f} | {speedup:<14} |")

    return results

if __name__ == "__main__":
    run_tts_benchmark()
