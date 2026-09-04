#!/usr/bin/env python3
"""
Alveare 3.0 Live Voice-to-Voice Terminal Client.

Connects to the Alveare Control Server WebSocket (/ws/live),
captures microphone input via native Linux audio (`arecord`),
and streams responses to speakers (`aplay`) with real-time HUD telemetry.
"""
import os
import sys
import time
import json
import base64
import signal
import asyncio
import subprocess
import tempfile
from pathlib import Path
import httpx
import websockets

def print_banner():
    print("\033[1;36m" + "=" * 65 + "\033[0m")
    print("\033[1;32m   ALVEARE 3.0 — LIVE FULL-DUPLEX VOICE-TO-VOICE STUDIO\033[0m")
    print("\033[1;36m" + "=" * 65 + "\033[0m")

async def main():
    print_banner()
    host = os.getenv("ALVEARE_HOST", "127.0.0.1")
    port = int(os.getenv("ALVEARE_PORT", "8080"))
    base_url = f"http://{host}:{port}"
    ws_url = f"ws://{host}:{port}/ws/live"

    print(f"\033[0;33m[1/3] Connecting to Alveare Control Center at {base_url}...\033[0m")
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            resp = await client.get(f"{base_url}/api/status")
            status_data = resp.json()
        except Exception:
            print(f"\033[1;31mError: Alveare server is not running on {base_url}.\033[0m")
            print("Start the control center first in another terminal with:")
            print("    ./alveare start")
            sys.exit(1)

    print(f"\033[0;33m[2/3] Checking Tri-Hardware Live Profile...\033[0m")
    slots = status_data.get("slots", {})
    llm_slot = slots.get("llm", {})
    stt_slot = slots.get("stt", {})
    tts_slot = slots.get("tts", {})

    print(f"  • LLM Chat:   \033[1;35m{llm_slot.get('model', 'gemma3')} on {llm_slot.get('device', 'gpu').upper()}\033[0m (status: {llm_slot.get('status', 'unknown')})")
    print(f"  • STT Voice:  \033[1;35m{stt_slot.get('model', 'whisper-base')} on {stt_slot.get('device', 'npu').upper()}\033[0m (status: {stt_slot.get('status', 'unknown')})")
    print(f"  • TTS Voice:  \033[1;35m{tts_slot.get('model', 'audio8-0.1b')} on {tts_slot.get('device', 'cpu').upper()}\033[0m (status: {tts_slot.get('status', 'unknown')})")

    # If any slot is stopped, start live profile automatically
    if llm_slot.get("status") != "running" or stt_slot.get("status") != "running" or tts_slot.get("status") != "running":
        print(f"\033[0;34m>>> Activating Tri-Hardware Live Session via /api/live/start...\033[0m")
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(f"{base_url}/api/live/start", json={
                "llm_model": "gemma3",
                "llm_device": "gpu",
                "stt_model": "openai/whisper-base",
                "stt_device": "npu",
                "tts_model": "Audio8/Audio8-TTS-Preview-0.1b",
                "tts_device": "cpu"
            })

    print(f"\033[0;33m[3/3] Opening WebSocket Live Stream to {ws_url}...\033[0m")
    try:
        async with websockets.connect(ws_url) as ws:
            print("\033[1;32m[LIVE] Connected! Speak into your microphone (press Ctrl+C to exit).\033[0m")
            print("-" * 65)

            current_player: list = [None]  # stores active aplay process

            def stop_playback():
                if current_player[0] and current_player[0].poll() is None:
                    try:
                        current_player[0].terminate()
                    except Exception:
                        pass
                    current_player[0] = None

            # 1. Background task to receive server events
            async def receiver():
                async for raw_msg in ws:
                    try:
                        ev = json.loads(raw_msg)
                        etype = ev.get("event")

                        if etype == "live_connected":
                            hp = ev.get("hardware_profile", {})
                            print(f"\033[1;34m[Live Engine Ready]\033[0m Tri-Hardware active:")
                            print(f"   LLM: {hp.get('llm', {}).get('model')} [{hp.get('llm', {}).get('device')}]")
                            print(f"   STT: {hp.get('stt', {}).get('model')} [{hp.get('stt', {}).get('device')}]")
                            print(f"   TTS: {hp.get('tts', {}).get('model')} [{hp.get('tts', {}).get('device')}]")
                        elif etype == "vad_speech_start":
                            stop_playback()
                            print("\n\033[1;33m● [User Speaking...]\033[0m", end="", flush=True)
                        elif etype == "interrupted":
                            stop_playback()
                            print("\n\033[1;31m⚡ [Barge-in Interruption Detected! Audio stopped]\033[0m")
                        elif etype == "user_transcript":
                            print(f"\r\033[1;37mUser:\033[0m {ev.get('text')} \033[0;36m(STT: {ev.get('stt_latency_ms')} ms)\033[0m")
                        elif etype == "ttft":
                            print(f"\033[0;35m[TTFT: {ev.get('ttft_ms')} ms]\033[0m \033[1;32mAlveare:\033[0m ", end="", flush=True)
                        elif etype == "llm_chunk":
                            print(f"{ev.get('text')} ", end="", flush=True)
                        elif etype == "audio_chunk":
                            b64 = ev.get("audio_b64")
                            if b64:
                                audio_bytes = base64.b64decode(b64)
                                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                                    tmp.write(audio_bytes)
                                    tmp_path = tmp.name

                                # Play audio via aplay in background
                                stop_playback()
                                current_player[0] = subprocess.Popen(
                                    ["aplay", "-q", tmp_path],
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL
                                )
                        elif etype == "turn_complete":
                            metrics = ev.get("metrics", {})
                            print(f"\n\033[0;36m[Turn Turnaround: {metrics.get('e2e_latency_ms')} ms | TTFT: {metrics.get('ttft_ms')} ms | TTFA: {metrics.get('ttfa_ms')} ms]\033[0m")
                            print("-" * 65)
                        elif etype == "error":
                            print(f"\n\033[1;31m[Error] {ev.get('error')}\033[0m")
                    except Exception as e:
                        pass

            # 2. Background task to stream microphone audio
            async def sender():
                arecord_bin = subprocess.run(["which", "arecord"], capture_output=True, text=True).stdout.strip()
                if not arecord_bin:
                    print("\033[1;31m[Notice] 'arecord' not found on system. Mic streaming disabled.\033[0m")
                    return

                # Record 16kHz 16-bit mono raw PCM
                cmd = [arecord_bin, "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "raw", "-q"]
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

                chunk_size = 1024 * 2  # ~64ms of audio
                loop = asyncio.get_running_loop()

                try:
                    while True:
                        data = await loop.run_in_executor(None, proc.stdout.read, chunk_size)
                        if not data:
                            break
                        await ws.send(data)
                finally:
                    proc.terminate()

            t_recv = asyncio.create_task(receiver())
            t_send = asyncio.create_task(sender())
            await asyncio.gather(t_recv, t_send)

    except (websockets.exceptions.ConnectionClosed, asyncio.CancelledError):
        print("\n\033[0;33m[Live] Disconnected.\033[0m")
    except KeyboardInterrupt:
        print("\n\033[0;33m[Live] Session ended by user.\033[0m")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
