"""
Alveare 3.0 Live Voice-to-Voice Orchestrator.

Full-duplex real-time conversational engine:
- Continuous audio streaming & Voice Activity Detection (VAD)
- Sub-second turn-around latency
- User barge-in interruption detection
- Streaming LLM token generation with sentence chunking
- Parallel TTS audio chunk synthesis & streaming playback
- Tri-hardware routing (e.g. LLM on GPU, STT on NPU, TTS on CPU)
"""
import os
import sys
import io
import time
import json
import uuid
import re
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Awaitable
import soundfile as sf
import httpx

from runtime.py.whisper_stt import WhisperSTT
from runtime.py.audio8_tts import Audio8TTS, split_into_chunks

class LiveTurnMetrics:
    def __init__(self):
        self.speech_start: float = 0.0
        self.speech_end: float = 0.0
        self.stt_start: float = 0.0
        self.stt_latency_ms: float = 0.0
        self.llm_start: float = 0.0
        self.ttft_ms: float = 0.0  # Time to First Token
        self.ttfa_ms: float = 0.0  # Time to First Audio chunk
        self.tts_total_ms: float = 0.0
        self.e2e_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "stt_latency_ms": round(self.stt_latency_ms, 1),
            "ttft_ms": round(self.ttft_ms, 1),
            "ttfa_ms": round(self.ttfa_ms, 1),
            "tts_total_ms": round(self.tts_total_ms, 1),
            "e2e_latency_ms": round(self.e2e_latency_ms, 1),
        }

class LiveSession:
    """
    Manages a single full-duplex live session.
    """
    def __init__(
        self,
        session_id: str,
        send_event_fn: Callable[[Dict[str, Any]], Awaitable[None]],
        llm_host: str = "127.0.0.1",
        llm_port: int = 8000,
        llm_model: str = "gemma3",
        stt_device: str = "npu",
        tts_device: str = "cpu",
        stt_model: str = "openai/whisper-base",
        tts_model: str = "Audio8/Audio8-TTS-Preview-0.1b",
        voice: str = "valeria",
        system_prompt: Optional[str] = None
    ):
        self.session_id = session_id
        self.send_event = send_event_fn
        self.llm_host = llm_host
        self.llm_port = llm_port
        self.llm_model = llm_model
        self.stt_device = stt_device
        self.tts_device = tts_device
        self.stt_model = stt_model
        self.tts_model = tts_model
        self.voice = voice

        # VAD Parameters
        self.sample_rate = 16000
        self.bytes_per_sample = 2  # 16-bit mono PCM
        self.speech_energy_threshold = 0.012  # RMS threshold
        self.min_speech_duration_s = 0.25
        self.min_silence_duration_s = 0.45
        self.max_turn_duration_s = 15.0

        # State
        self.turn_id: int = 0
        self.is_ai_speaking: bool = False
        self.audio_buffer: bytearray = bytearray()
        self.speech_buffer: bytearray = bytearray()
        self.in_speech: bool = False
        self.silence_samples_count: int = 0
        self.speech_samples_count: int = 0
        self.speech_start_time: float = 0.0

        default_prompt = (
            "Sei Alveare Live, un assistente vocale AI conversazionale ultra-veloce ed empatico che gira su hardware AMD Ryzen AI (NPU, GPU Radeon, CPU). "
            "Rispondi in modo diretto, conciso, naturale e colloquiale (1-3 frasi al massimo per risposta, senza elenchi puntati o markdown complicato). "
            "Parla in italiano naturale come in una vera telefonata o conversazione a voce."
        )
        self.history: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt or default_prompt}
        ]

        # Active background processing task for current turn
        self.current_processing_task: Optional[asyncio.Task] = None

        # Workers
        self.stt: Optional[WhisperSTT] = None
        self.tts: Optional[Audio8TTS] = None

    def initialize_workers(self):
        """Pre-warm or reference STT and TTS singletons."""
        self.stt = WhisperSTT.get_instance(model_id=self.stt_model, device=self.stt_device)
        self.tts = Audio8TTS.get_instance(model_id=self.tts_model, device=self.tts_device)

    async def handle_audio_chunk(self, chunk: bytes):
        """
        Process incoming audio chunk (16-bit 16kHz PCM).
        Performs real-time VAD, barge-in interruption detection, and turn triggering.
        """
        if not chunk:
            return

        # Convert int16 PCM to float32 numpy array for RMS energy calculation
        try:
            samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception:
            return

        if len(samples) == 0:
            return

        rms = float(np.sqrt(np.mean(samples ** 2)))
        chunk_duration_s = len(samples) / self.sample_rate

        is_voice = rms > self.speech_energy_threshold

        # --- BARGE-IN INTERRUPTION HANDLING ---
        if is_voice and self.is_ai_speaking:
            # User cut in while assistant was speaking!
            self.is_ai_speaking = False
            self.turn_id += 1
            if self.current_processing_task and not self.current_processing_task.done():
                self.current_processing_task.cancel()
                self.current_processing_task = None

            await self.send_event({
                "event": "interrupted",
                "turn_id": self.turn_id,
                "message": "User interruption detected (barge-in)",
                "timestamp": time.time()
            })

        # --- VAD STATE MACHINE ---
        if is_voice:
            if not self.in_speech:
                self.in_speech = True
                self.speech_buffer = bytearray()
                self.speech_start_time = time.time()
                self.speech_samples_count = 0
                await self.send_event({
                    "event": "vad_speech_start",
                    "timestamp": self.speech_start_time
                })

            self.speech_buffer.extend(chunk)
            self.speech_samples_count += len(samples)
            self.silence_samples_count = 0

            # Safeguard max turn duration
            if (self.speech_samples_count / self.sample_rate) > self.max_turn_duration_s:
                await self._trigger_turn_completion()
        else:
            if self.in_speech:
                self.speech_buffer.extend(chunk)
                self.silence_samples_count += len(samples)
                silence_duration_s = self.silence_samples_count / self.sample_rate

                if silence_duration_s >= self.min_silence_duration_s:
                    speech_duration_s = self.speech_samples_count / self.sample_rate
                    if speech_duration_s >= self.min_speech_duration_s:
                        await self._trigger_turn_completion()
                    else:
                        # Too short (click, breath, or background tap) -> discard
                        self.in_speech = False
                        self.speech_buffer = bytearray()
                        self.silence_samples_count = 0
                        self.speech_samples_count = 0

    async def _trigger_turn_completion(self):
        """End of user speech detected, trigger processing."""
        self.in_speech = False
        speech_pcm = bytes(self.speech_buffer)
        self.speech_buffer = bytearray()
        self.silence_samples_count = 0
        self.speech_samples_count = 0

        self.turn_id += 1
        turn_id = self.turn_id

        await self.send_event({
            "event": "vad_speech_end",
            "turn_id": turn_id,
            "timestamp": time.time()
        })

        # Cancel prior turn if still running
        if self.current_processing_task and not self.current_processing_task.done():
            self.current_processing_task.cancel()

        self.current_processing_task = asyncio.create_task(
            self._process_voice_turn(speech_pcm, turn_id)
        )

    async def _process_voice_turn(self, pcm_data: bytes, turn_id: int):
        """
        Processes a complete user turn:
        1. STT: transcribe user audio -> text
        2. LLM: stream reasoning response
        3. Sentence Chunker: split streaming text into natural speech clauses
        4. TTS: synthesize each sentence chunk to audio
        5. Stream audio packets back to client
        """
        metrics = LiveTurnMetrics()
        turn_start = time.perf_counter()

        # Build WAV in memory for STT
        try:
            samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
            wav_io = io.BytesIO()
            sf.write(wav_io, samples, self.sample_rate, format="WAV", subtype="PCM_16")
            wav_bytes = wav_io.getvalue()
        except Exception as e:
            await self.send_event({"event": "error", "error": f"PCM conversion failed: {e}"})
            return

        # 1. Speech-to-Text
        metrics.stt_start = time.perf_counter()
        if not self.stt:
            self.initialize_workers()

        stt_res = await asyncio.to_thread(self.stt.transcribe, wav_bytes)
        metrics.stt_latency_ms = (time.perf_counter() - metrics.stt_start) * 1000.0

        if turn_id != self.turn_id:
            return  # Interrupted during STT

        user_text = stt_res.get("text", "").strip()

        # Filter hallucinations or empty silence
        cleaned = re.sub(r'\[.*?\]', '', user_text).strip()
        if not cleaned or len(cleaned) < 2:
            return

        await self.send_event({
            "event": "user_transcript",
            "text": user_text,
            "turn_id": turn_id,
            "stt_latency_ms": round(metrics.stt_latency_ms, 1)
        })

        self.history.append({"role": "user", "content": user_text})
        self.is_ai_speaking = True

        # 2. LLM Streaming
        metrics.llm_start = time.perf_counter()
        target_url = f"http://{self.llm_host}:{self.llm_port}/v1/chat/completions"
        payload = {
            "model": self.llm_model,
            "messages": self.history,
            "stream": True,
            "max_tokens": 300,
            "temperature": 0.7,
        }

        full_assistant_text = ""
        sentence_buffer = ""
        sentence_index = 0
        first_token_received = False

        sentence_delimiters = re.compile(r'([.!?;\n:…]+\s*)')

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", target_url, json=payload) as resp:
                    if resp.status_code != 200:
                        err_body = await resp.aread()
                        await self.send_event({
                            "event": "error",
                            "error": f"LLM error {resp.status_code}: {err_body.decode('utf-8', 'replace')}"
                        })
                        self.is_ai_speaking = False
                        return

                    async for line in resp.aiter_lines():
                        if turn_id != self.turn_id:
                            # User interrupted! Abort LLM stream
                            break

                        if not line or not line.startswith("data: "):
                            continue

                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break

                        try:
                            delta_json = json.loads(data_str)
                            choices = delta_json.get("choices", [])
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})
                            content_piece = delta.get("content", "")
                        except Exception:
                            continue

                        if not content_piece:
                            continue

                        if not first_token_received:
                            first_token_received = True
                            metrics.ttft_ms = (time.perf_counter() - metrics.llm_start) * 1000.0
                            await self.send_event({
                                "event": "ttft",
                                "ttft_ms": round(metrics.ttft_ms, 1),
                                "turn_id": turn_id
                            })

                        full_assistant_text += content_piece
                        sentence_buffer += content_piece

                        # Check for sentence boundaries to yield audio chunks early
                        splits = sentence_delimiters.split(sentence_buffer)
                        if len(splits) >= 3:
                            # At least one complete sentence + remaining buffer
                            ready_sentence = (splits[0] + splits[1]).strip()
                            sentence_buffer = "".join(splits[2:])

                            if ready_sentence and len(ready_sentence) > 3:
                                sentence_index += 1
                                await self._synthesize_and_stream_chunk(
                                    ready_sentence, sentence_index, turn_id, metrics, turn_start
                                )

                    # End of LLM stream: flush remaining buffer
                    if turn_id == self.turn_id and sentence_buffer.strip():
                        sentence_index += 1
                        await self._synthesize_and_stream_chunk(
                            sentence_buffer.strip(), sentence_index, turn_id, metrics, turn_start
                        )

        except asyncio.CancelledError:
            self.is_ai_speaking = False
            return
        except Exception as e:
            if turn_id == self.turn_id:
                await self.send_event({"event": "error", "error": f"Live LLM generation failed: {e}"})
            self.is_ai_speaking = False
            return

        if turn_id == self.turn_id:
            self.history.append({"role": "assistant", "content": full_assistant_text.strip()})
            metrics.e2e_latency_ms = (time.perf_counter() - turn_start) * 1000.0

            await self.send_event({
                "event": "turn_complete",
                "turn_id": turn_id,
                "full_text": full_assistant_text.strip(),
                "metrics": metrics.to_dict(),
                "timestamp": time.time()
            })
            self.is_ai_speaking = False

    async def _synthesize_and_stream_chunk(
        self,
        text_chunk: str,
        chunk_idx: int,
        turn_id: int,
        metrics: LiveTurnMetrics,
        turn_start: float
    ):
        """Synthesize a text chunk to speech and stream audio to client immediately."""
        if turn_id != self.turn_id:
            return

        t_chunk_start = time.perf_counter()
        await self.send_event({
            "event": "llm_chunk",
            "chunk_idx": chunk_idx,
            "text": text_chunk,
            "turn_id": turn_id
        })

        if not self.tts:
            self.initialize_workers()

        tts_res = await asyncio.to_thread(
            self.tts.generate,
            text=text_chunk,
            voice=self.voice,
            max_new_tokens=250,
            temperature=0.8
        )

        if turn_id != self.turn_id:
            return

        chunk_lat = (time.perf_counter() - t_chunk_start) * 1000.0
        metrics.tts_total_ms += chunk_lat

        if metrics.ttfa_ms == 0.0:
            metrics.ttfa_ms = (time.perf_counter() - turn_start) * 1000.0

        if tts_res.get("status") == "success":
            audio_path = tts_res.get("audio_path")
            audio_b64 = ""
            if audio_path and os.path.exists(audio_path):
                with open(audio_path, "rb") as af:
                    audio_b64 = base64.b64encode(af.read()).decode("utf-8")

            await self.send_event({
                "event": "audio_chunk",
                "chunk_idx": chunk_idx,
                "audio_b64": audio_b64,
                "text": text_chunk,
                "turn_id": turn_id,
                "duration_sec": tts_res.get("duration_sec", 0.0),
                "ttfa_ms": round(metrics.ttfa_ms, 1),
                "chunk_latency_ms": round(chunk_lat, 1)
            })
        else:
            await self.send_event({
                "event": "tts_error",
                "chunk_idx": chunk_idx,
                "turn_id": turn_id,
                "error": tts_res.get("error", "Unknown TTS error")
            })
