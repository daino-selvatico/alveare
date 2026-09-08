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
import base64
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Awaitable
import soundfile as sf
import httpx

from runtime.py.whisper_stt import WhisperSTT
from runtime.py.tts_factory import get_tts_engine, is_kokoro_model, split_into_chunks
from runtime.py.kokoro_tts import KokoroTTS
from runtime.py.audio8_tts import Audio8TTS

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

class StreamingThoughtFilter:
    """
    Real-time streaming filter that detects and removes reasoning/thought blocks
    such as <thought>...</thought>, <think>...</think>, and channel tokens
    before tokens reach UI or TTS.
    """
    def __init__(self):
        self.in_thought = False
        self.tag_buffer = ""
        self.open_tag_pattern = re.compile(
            r'<thought\b[^>]*>|<think\b[^>]*>|<\|thought\|>|<\|channel\|>thought|<\|channel>thought',
            re.IGNORECASE
        )
        self.close_tag_pattern = re.compile(
            r'</thought>|</think>|<\|/thought\|>|<channel\|>|<\|channel\|>',
            re.IGNORECASE
        )

    def filter(self, chunk: str) -> str:
        if not chunk:
            return ""

        text = self.tag_buffer + chunk
        self.tag_buffer = ""
        out = []

        while text:
            if self.in_thought:
                m_close = self.close_tag_pattern.search(text)
                if m_close:
                    self.in_thought = False
                    text = text[m_close.end():]
                else:
                    partial = re.search(r'</?[a-zA-Z0-9_|]*$', text)
                    if partial and len(partial.group(0)) < 15:
                        self.tag_buffer = partial.group(0)
                        text = text[:partial.start()]
                    break
            else:
                m_open = self.open_tag_pattern.search(text)
                if m_open:
                    out.append(text[:m_open.start()])
                    self.in_thought = True
                    text = text[m_open.end():]
                else:
                    partial = re.search(r'<[a-zA-Z0-9_|]*$', text)
                    if partial and len(partial.group(0)) < 20:
                        out.append(text[:partial.start()])
                        self.tag_buffer = partial.group(0)
                        break
                    else:
                        out.append(text)
                        break

        filtered = "".join(out)
        filtered = self.open_tag_pattern.sub("", filtered)
        filtered = self.close_tag_pattern.sub("", filtered)
        return filtered

    def flush(self) -> str:
        if not self.in_thought and self.tag_buffer:
            res = self.tag_buffer
            self.tag_buffer = ""
            return res
        self.tag_buffer = ""
        return ""

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
        llm_model: str = "gemma4-e4b",
        stt_device: str = "cpu",
        tts_device: str = "cpu",
        stt_model: str = "openai/whisper-large-v3-turbo",
        tts_model: str = "hexgrad/Kokoro-82M",
        voice: str = "if_sara",
        language: str = "it",
        system_prompt: Optional[str] = None,
        max_context_turns: int = 4,
        max_tokens: int = 120,
        temperature: float = 0.5,
        speed: float = 1.0,
        enable_thinking: bool = False
    ):
        if tts_model and is_kokoro_model(tts_model):
            if not tts_device or tts_device == "npu":
                tts_device = "cpu"
            if voice in ("valeria", "female_expressive", "default"):
                voice = "if_sara"

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
        self.language = language or "it"

        # Tunable Generation & History Parameters
        self.max_context_turns = int(max_context_turns) if max_context_turns is not None else 4
        self.max_tokens = int(max_tokens) if max_tokens is not None else 120
        self.temperature = float(temperature) if temperature is not None else 0.5
        self.speed = float(speed) if speed is not None else 1.0
        self.enable_thinking = bool(enable_thinking)

        # VAD Parameters
        self.sample_rate = 16000
        self.bytes_per_sample = 2  # 16-bit mono PCM
        self.speech_energy_threshold = 0.015  # Base RMS threshold
        self.noise_floor = 0.005  # Adaptive ambient noise floor estimate
        self.min_speech_duration_s = 0.25
        self.min_silence_duration_s = 0.35  # Reduced from 0.45s for snappier turn-taking
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
            "Sei Alveare Live, assistente vocale ultra-rapido. Rispondi SEMPRE in modo brevissimo, diretto e conciso (massimo 1-2 frasi brevi, senza preamboli, senza elenchi). Parla in italiano naturale."
        )
        self.history: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt or default_prompt}
        ]

        # Active background processing task for current turn
        self.current_processing_task: Optional[asyncio.Task] = None

    def update_settings(self, settings: Dict[str, Any]):
        """Dynamically updates live session settings without restarting the connection."""
        if not settings or not isinstance(settings, dict):
            return

        if "max_context_turns" in settings:
            try:
                self.max_context_turns = max(1, min(100, int(settings["max_context_turns"])))
            except (ValueError, TypeError):
                pass

        if "max_tokens" in settings:
            try:
                self.max_tokens = max(20, min(1024, int(settings["max_tokens"])))
            except (ValueError, TypeError):
                pass

        if "temperature" in settings:
            try:
                self.temperature = max(0.0, min(2.0, float(settings["temperature"])))
            except (ValueError, TypeError):
                pass

        if "voice" in settings and settings["voice"]:
            v = str(settings["voice"]).strip()
            if self.tts_model and is_kokoro_model(self.tts_model):
                if v in ("valeria", "female_expressive", "default"):
                    v = "if_sara"
            self.voice = v

        if "speed" in settings:
            try:
                self.speed = max(0.5, min(2.0, float(settings["speed"])))
            except (ValueError, TypeError):
                pass

        if "system_prompt" in settings and settings["system_prompt"] is not None:
            prompt_text = str(settings["system_prompt"]).strip()
            if prompt_text:
                if self.history and self.history[0].get("role") == "system":
                    self.history[0]["content"] = prompt_text
                else:
                    self.history.insert(0, {"role": "system", "content": prompt_text})

        if "min_silence_duration_s" in settings:
            try:
                self.min_silence_duration_s = max(0.15, min(2.0, float(settings["min_silence_duration_s"])))
            except (ValueError, TypeError):
                pass

        if "enable_thinking" in settings:
            self.enable_thinking = bool(settings["enable_thinking"])

        # Workers
        self.stt: Optional[WhisperSTT] = None
        self.tts: Optional[Any] = None

    def initialize_workers(self):
        """Pre-warm or reference STT and TTS singletons."""
        self.stt = WhisperSTT.get_instance(model_id=self.stt_model, device=self.stt_device)
        self.tts = get_tts_engine(model_id=self.tts_model, device=self.tts_device)

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

        # Adaptive threshold calibration based on background ambient noise
        if not self.in_speech:
            self.noise_floor = 0.95 * self.noise_floor + 0.05 * rms

        effective_threshold = max(self.speech_energy_threshold, self.noise_floor * 2.2)
        is_voice = rms > effective_threshold

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
            if len(samples) < int(self.sample_rate * self.min_speech_duration_s):
                return
            rms = float(np.sqrt(np.mean(samples ** 2)))
            peak = float(np.max(np.abs(samples))) if len(samples) > 0 else 0.0
            # Filter audio too silent or background noise/tap
            if rms < self.speech_energy_threshold * 0.7 or peak < 0.03:
                return

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

        stt_res = await asyncio.to_thread(self.stt.transcribe, wav_bytes, language=self.language)
        metrics.stt_latency_ms = (time.perf_counter() - metrics.stt_start) * 1000.0

        if turn_id != self.turn_id:
            return  # Interrupted during STT

        user_text = stt_res.get("text", "").strip()

        # Filter hallucinations or empty silence
        cleaned = re.sub(r'\[.*?\]|\(.*?\)', '', user_text).strip()
        noise_phrases = {
            "grazie", "grazie per la visione", "grazie per l'attenzione",
            "grazie mille", "sottotitoli", "sottotitoli creati dalla comunità",
            "you", "thank you", "thanks for watching", "bye", "ciao",
            "...", ".", ""
        }
        if not cleaned or len(cleaned) < 2:
            return
        if cleaned.lower().strip(".!? ") in noise_phrases and rms < self.speech_energy_threshold * 1.3:
            return

        await self.send_event({
            "event": "user_transcript",
            "text": user_text,
            "turn_id": turn_id,
            "stt_latency_ms": round(metrics.stt_latency_ms, 1)
        })

        self.history.append({"role": "user", "content": user_text})
        self.is_ai_speaking = True

        # Truncate / window history based on max_context_turns
        system_msg = self.history[0] if self.history and self.history[0].get("role") == "system" else None
        conversation_msgs = self.history[1:] if system_msg else self.history
        max_msgs = max(2, self.max_context_turns * 2)
        pruned_msgs = conversation_msgs[-max_msgs:] if len(conversation_msgs) > max_msgs else conversation_msgs
        messages_to_send = ([system_msg] if system_msg else []) + pruned_msgs

        # 2. LLM Streaming
        metrics.llm_start = time.perf_counter()
        target_url = f"http://{self.llm_host}:{self.llm_port}/v1/chat/completions"
        payload = {
            "model": self.llm_model,
            "messages": messages_to_send,
            "stream": True,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "enable_thinking": self.enable_thinking,
        }

        full_assistant_text = ""
        sentence_buffer = ""
        sentence_index = 0
        first_token_received = False
        thought_filter = StreamingThoughtFilter()

        delimiter_pattern = re.compile(r'[,;:.?!\n…]+')

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

                        # Filter out thought/reasoning blocks in real-time
                        filtered_piece = thought_filter.filter(content_piece)
                        if not filtered_piece:
                            continue

                        if not first_token_received:
                            first_token_received = True
                            metrics.ttft_ms = (time.perf_counter() - metrics.llm_start) * 1000.0
                            await self.send_event({
                                "event": "ttft",
                                "ttft_ms": round(metrics.ttft_ms, 1),
                                "turn_id": turn_id
                            })

                        full_assistant_text += filtered_piece
                        sentence_buffer += filtered_piece

                        # Ultra-fast aggressive clause streaming pipelining:
                        # 1. Split on any punctuation (, ; : . ? ! \n …) as soon as >= 3 words or > 15 characters
                        # 2. Or split if buffer exceeds 6 words even without punctuation
                        while True:
                            ready_sentence = ""
                            for m in delimiter_pattern.finditer(sentence_buffer):
                                cand = sentence_buffer[:m.end()].strip()
                                words = cand.split()
                                if len(words) >= 3 or len(cand) > 15:
                                    ready_sentence = cand
                                    sentence_buffer = sentence_buffer[m.end():].lstrip()
                                    break

                            if not ready_sentence:
                                word_matches = list(re.finditer(r'\S+', sentence_buffer))
                                if len(word_matches) > 6:
                                    match_6th = word_matches[5]
                                    ready_sentence = sentence_buffer[:match_6th.end()].strip()
                                    sentence_buffer = sentence_buffer[match_6th.end():].lstrip()

                            if ready_sentence:
                                sentence_index += 1
                                await self._synthesize_and_stream_chunk(
                                    ready_sentence, sentence_index, turn_id, metrics, turn_start
                                )
                            else:
                                break

                    # Flush any pending tag buffer from thought filter
                    thought_tail = thought_filter.flush()
                    if thought_tail:
                        full_assistant_text += thought_tail
                        sentence_buffer += thought_tail

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
            speed=getattr(self, "speed", 1.0),
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
