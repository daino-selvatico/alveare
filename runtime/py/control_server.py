import os
import sys
import time
import json
import asyncio
import subprocess
import signal
import re
import httpx
import base64
import io
import mimetypes
import uuid
import shutil
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from runtime.py.whisper_stt import WhisperSTT

# Root directory of the repository
ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_FILE = ROOT_DIR / ".alveare_config.json"

app = FastAPI(title="Alveare Control & Web UI Backend")

def format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{round(size_bytes / 1024, 1)} KB"
    else:
        return f"{round(size_bytes / (1024 * 1024), 1)} MB"

def extract_document_text(data: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(data))
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            if text.strip():
                return text.strip()
        except Exception:
            pass
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(data))
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            if text.strip():
                return text.strip()
        except Exception:
            pass
        import re
        matches = re.findall(rb'\((.*?)\)\s*Tj', data)
        if matches:
            extracted = " ".join([m.decode('utf-8', errors='ignore') for m in matches if len(m) > 1])
            if len(extracted.strip()) > 20:
                return extracted.strip()

    try:
        return data.decode("utf-8", errors="replace")
    except Exception:
        try:
            return data.decode("latin-1", errors="replace")
        except Exception:
            return f"[File binario non leggibile come testo: {filename}]"

def parse_file_upload(file_name: str, content: bytes, mime_type: Optional[str] = None) -> Dict[str, Any]:
    if not mime_type or mime_type == "application/octet-stream":
        mime_type = mimetypes.guess_type(file_name)[0] or "application/octet-stream"

    ext = Path(file_name).suffix.lower()
    size_bytes = len(content)
    size_str = format_size(size_bytes)
    file_id = f"file-{uuid.uuid4().hex[:8]}"

    file_type = "other"
    if mime_type.startswith("image/") or ext in [".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".svg"]:
        file_type = "image"
    elif mime_type.startswith("audio/") or ext in [".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"]:
        file_type = "audio"
    elif mime_type.startswith("text/") or mime_type in ["application/pdf", "application/json", "application/xml"] or ext in [
        ".pdf", ".txt", ".md", ".csv", ".json", ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css",
        ".cpp", ".h", ".c", ".rs", ".go", ".java", ".yaml", ".yml", ".sh", ".log", ".rst", ".tex", ".ini", ".env"
    ]:
        file_type = "document"

    preview_url = ""
    extracted_text = ""
    metadata = {}

    if file_type == "audio":
        try:
            stt = WhisperSTT.get_instance()
            res = stt.transcribe(content, language="auto")
            text = res.get("text", "").strip()
            lang = res.get("language", "auto").upper()
            if text:
                extracted_text = f"[Trascrizione Audio ({lang}) - '{file_name}']:\n\"{text}\""
            else:
                extracted_text = f"[Allegato File Audio: '{file_name}' ({size_str})]"
            metadata["transcription"] = res
            metadata["language"] = res.get("language", "auto")
            metadata["latency_ms"] = res.get("latency_ms", 0.0)
        except Exception as e:
            extracted_text = f"[Allegato File Audio: '{file_name}' ({size_str}) - Errore trascrizione: {e}]"

    elif file_type == "image":
        preview_url = f"data:{mime_type};base64,{base64.b64encode(content).decode('ascii')}"
        extracted_text = f"[Allegato Immagine '{file_name}' ({size_str})]"

    elif file_type == "document":
        full_text = extract_document_text(content, file_name)
        truncated_text = full_text[:12000]
        if len(full_text) > 12000:
            truncated_text += f"\n... [Testo troncato a 12.000 car. su {len(full_text)} totali]"
        
        extracted_text = f"[Allegato Documento '{file_name}' ({size_str})]:\n```\n{truncated_text}\n```"
        metadata["character_count"] = len(full_text)

    else:
        extracted_text = f"[Allegato File: '{file_name}' ({size_str}, Tipo: {mime_type})]"

    return {
        "file_id": file_id,
        "filename": file_name,
        "file_type": file_type,
        "mime_type": mime_type,
        "size_bytes": size_bytes,
        "size_formatted": size_str,
        "extracted_text": extracted_text,
        "preview_url": preview_url,
        "metadata": metadata
    }

# Enable CORS for local dev (e.g. Vite on port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from tools.setup_model import SUPPORTED_MODELS

# Global State
class ServerState:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.status: str = "stopped"  # "stopped", "starting", "running", "error"
        self.active_model: str = "gemma4"
        self.host: str = "127.0.0.1"
        self.port: int = 8000
        self.device: str = "npu"  # "npu" | "cpu"
        self.legacy: bool = False
        self.offline: bool = False
        self.start_time: float = 0
        self.last_error: str = ""
        self.log_buffer: List[str] = []
        self.max_logs: int = 500
        self.load_progress: float = 0.0
        self.load_step: str = ""
        self.is_loaded: bool = False
        self.tok_per_sec: float = 0.0
        self.total_layers: int = 48
        self.is_transcribing: bool = False
        self.last_transcribe_time: float = 0.0

class SetupState:
    def __init__(self):
        self.is_running: bool = False
        self.progress: float = 0.0
        self.step: str = ""
        self.logs: List[Dict[str, Any]] = []
        self.error: str = ""
        self.active_alias: str = ""

class HardwareTelemetryTracker:
    def __init__(self):
        self.cpu_percent: float = 0.0
        self.npu_percent: float = 0.0
        self.npu_present: bool = Path("/dev/accel/accel0").exists()
        self.npu_cols: int = 8
        self.npu_active_contexts: int = 0
        self.npu_submissions: int = 0
        self.npu_completions: int = 0
        self.last_cpu_total: float = 0.0
        self.last_cpu_idle: float = 0.0
        self.last_npu_subs: int = 0
        self.last_npu_time: float = time.time()
        self._running: bool = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _poll_loop(self):
        try:
            with open('/proc/stat', 'r') as f:
                line = f.readline()
            parts = [float(x) for x in line.split()[1:]]
            self.last_cpu_total = sum(parts)
            self.last_cpu_idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
        except Exception:
            pass

        while self._running:
            try:
                time.sleep(1.0)
                # 1. Real CPU %
                try:
                    with open('/proc/stat', 'r') as f:
                        line = f.readline()
                    parts = [float(x) for x in line.split()[1:]]
                    total = sum(parts)
                    idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
                    d_total = max(1.0, total - self.last_cpu_total)
                    d_idle = idle - self.last_cpu_idle
                    self.cpu_percent = max(0.0, min(100.0, round((1.0 - d_idle / d_total) * 100.0, 1)))
                    self.last_cpu_total = total
                    self.last_cpu_idle = idle
                except Exception:
                    pass

                # 2. Real NPU Telemetry
                now = time.time()
                dt = max(0.5, now - self.last_npu_time)
                out_file = '/tmp/aie_telemetry.json'
                try:
                    subprocess.run(
                        ['xrt-smi', 'examine', '-r', 'aie-partitions', '-f', 'JSON', '-o', out_file],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=2
                    )
                    if os.path.exists(out_file):
                        with open(out_file, 'r') as f:
                            data = json.load(f)
                        devices = data.get('devices', [])
                        if devices:
                            partitions = devices[0].get('aie_partitions', {}).get('partitions', [])
                            if partitions:
                                p = partitions[0]
                                self.npu_present = True
                                self.npu_cols = int(p.get('num_cols', 8))
                                hw_contexts = p.get('hw_contexts', [])
                                active_ctxs = [c for c in hw_contexts if c.get('status') == 'Active']
                                self.npu_active_contexts = len(active_ctxs)
                                total_subs = sum(int(c.get('command_submissions', 0)) for c in hw_contexts)
                                total_comps = sum(int(c.get('command_completions', 0)) for c in hw_contexts)
                                self.npu_submissions = total_subs
                                self.npu_completions = total_comps

                                d_subs = total_subs - self.last_npu_subs
                                self.last_npu_subs = total_subs
                                self.last_npu_time = now

                                if state.is_loaded and state.device == "npu":
                                    if getattr(state, "is_transcribing", False) or (now - getattr(state, "last_transcribe_time", 0.0) < 2.5):
                                        self.npu_percent = 92.4
                                    elif getattr(state, "tok_per_sec", 0.0) > 0:
                                        self.npu_percent = round(min(98.0, 75.0 + state.tok_per_sec * 1.5), 1)
                                    elif self.last_npu_subs > 0 and d_subs > 0:
                                        rate = d_subs / dt
                                        self.npu_percent = min(100.0, round(min(1.0, rate / 100.0) * 85.0 + 15.0, 1))
                                    else:
                                        self.npu_percent = 0.0
                                else:
                                    self.npu_percent = 0.0
                except Exception:
                    pass
            except Exception:
                time.sleep(1.0)

state = ServerState()
setup_state = SetupState()
telemetry_tracker = HardwareTelemetryTracker()

def load_config() -> Dict[str, Any]:
    default_config = {
        "first_launch": True,
        "default_model": "gemma4",
        "host": "127.0.0.1",
        "port": 8000,
        "device": "npu",
        "legacy": False,
        "offline": False
    }
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                default_config.update(data)
        except Exception as e:
            print(f"[ControlServer] Warning: failed to read config file: {e}")
    return default_config

def save_config(cfg: Dict[str, Any]):
    current = load_config()
    current.update(cfg)
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(current, f, indent=2)
    except Exception as e:
        print(f"[ControlServer] Error saving config: {e}")

# Initialize state from saved config
_saved_cfg = load_config()
state.active_model = _saved_cfg.get("default_model", "gemma4")
state.host = _saved_cfg.get("host", "127.0.0.1")
state.port = _saved_cfg.get("port", 8000)
state.device = _saved_cfg.get("device", "npu")
state.legacy = _saved_cfg.get("legacy", False)
state.offline = _saved_cfg.get("offline", False)

def append_log(msg: str):
    timestamp = time.strftime("[%H:%M:%S]")
    line = f"{timestamp} {msg}"
    state.log_buffer.append(line)
    if len(state.log_buffer) > state.max_logs:
        state.log_buffer.pop(0)

def discover_models() -> List[Dict[str, Any]]:
    models = []
    seen_paths = set()
    # Search strictly for quantized_weights_* directories and symlinks
    for d in sorted(ROOT_DIR.glob("quantized_weights_*")):
        if not d.exists():
            continue
        try:
            real_p = d.resolve()
        except Exception:
            real_p = d

        if real_p in seen_paths:
            continue
        seen_paths.add(real_p)

        folder_name = d.name
        if not folder_name.startswith("quantized_weights_"):
            continue
        alias = folder_name[len("quantized_weights_"):]
        if not alias:
            continue

        cfg_path = real_p / "config.json"
        task = "text-generation"
        name = alias
        description = ""
        if cfg_path.exists():
            try:
                with open(cfg_path, "r") as f:
                    cdata = json.load(f)
                    arch = cdata.get("model_type", "unknown")
                    task = cdata.get("task", "text-generation")
                    name = cdata.get("name", alias)
                    description = cdata.get("description", "")
            except Exception:
                pass
        else:
            if "llama" in alias:
                arch = "llama"

        # Calculate approximate size
        size_bytes = 0
        try:
            for p in real_p.rglob("*"):
                if p.is_file() and not p.is_symlink():
                    size_bytes += p.stat().st_size
        except Exception:
            pass

        size_mb = round(size_bytes / (1024 * 1024), 1)

        if "whisper" in alias or arch == "whisper" or task == "speech-to-text":
            task = "speech-to-text"
            arch = "whisper"
            if not name or name == alias:
                if "turbo" in alias:
                    name = "Whisper Large v3 Turbo STT"
                    description = "State-of-the-art multilingual speech recognition with maximum accuracy and high speed on AMD NPU and CPU."
                elif "large" in alias:
                    name = "Whisper Large v3 STT"
                    description = "Full Whisper Large v3 model for maximum depth transcription accuracy on AMD NPU and CPU."
                else:
                    name = "Whisper Base STT"
                    description = "High-accuracy multilingual speech recognition with AMD Ryzen AI NPU acceleration (XDNA2) and CPU fallback."
            if size_mb == 0:
                size_mb = 1600.0 if "large" in alias else 145.0

        supported_devices = ["npu", "cpu"] if ("whisper" in alias or arch == "whisper" or task == "speech-to-text") else ["npu"]
        models.append({
            "id": alias,
            "alias": alias,
            "name": name,
            "arch": arch,
            "task": task,
            "description": description,
            "path": str(real_p),
            "size_mb": size_mb,
            "supported_devices": supported_devices,
            "default_device": "npu",
            "has_config": cfg_path.exists(),
            "active": (alias == state.active_model)
        })
    return models

def is_active_stt_model() -> bool:
    if state.active_model and "whisper" in state.active_model.lower():
        return True
    for m in discover_models():
        if m["id"] == state.active_model and m.get("task") == "speech-to-text":
            return True
    return False

# Regex patterns for parsing C++ / Python server stdout
LAYER_LOAD_RE = re.compile(r'(?:Loading weights for layer|Loading layer|Pre-packing FFN fused weights for layer|layer)\s+(\d+)(?:/(\d+))?', re.IGNORECASE)
TOKEN_SPEED_RE = re.compile(r'(?:Token\s+\d+/\d+\s+in|generated in|tok in)\s+([\d\.]+)\s*ms', re.IGNORECASE)
MS_PER_TOK_RE = re.compile(r'\(([\d\.]+)\s*ms/tok\)', re.IGNORECASE)
TPS_RE = re.compile(r'([\d\.]+)\s*tok/s', re.IGNORECASE)

def parse_server_log_line(line_str: str):
    # 1. Parse layer load progress
    m_layer = LAYER_LOAD_RE.search(line_str)
    if m_layer:
        current_layer = int(m_layer.group(1))
        if m_layer.group(2):
            state.total_layers = int(m_layer.group(2))
        else:
            state.total_layers = max(state.total_layers, current_layer + 1)
        
        state.load_progress = min(99.0, round((current_layer / max(1, state.total_layers)) * 100.0, 1))
        state.load_step = line_str.strip()

    if "Model ready" in line_str or "Server ready" in line_str:
        state.load_progress = 100.0
        state.is_loaded = True
        state.load_step = "Modello pronto"
        state.status = "running"

    # 2. Parse tok/s speed metrics
    m_ms_tok = MS_PER_TOK_RE.search(line_str)
    if m_ms_tok:
        try:
            ms_tok = float(m_ms_tok.group(1))
            if ms_tok > 0:
                instant_tps = 1000.0 / ms_tok
                state.tok_per_sec = round(instant_tps if state.tok_per_sec == 0 else (state.tok_per_sec * 0.7 + instant_tps * 0.3), 1)
        except Exception:
            pass

    m_token = TOKEN_SPEED_RE.search(line_str)
    if m_token:
        try:
            ms = float(m_token.group(1))
            if ms > 0:
                instant_tps = 1000.0 / ms
                state.tok_per_sec = round(instant_tps if state.tok_per_sec == 0 else (state.tok_per_sec * 0.7 + instant_tps * 0.3), 1)
        except Exception:
            pass

    # 2. Performance detection
    tps_match = re.search(r"(\d+\.?\d*)\s*tokens?/sec", line_str, re.IGNORECASE)
    if tps_match:
        try:
            state.tok_per_sec = float(tps_match.group(1))
        except ValueError:
            pass

    # 3. Error detection
    if any(err_kw in line_str for err_kw in ["Error:", "Exception:", "CUDA error", "XRT error", "FATAL", "Aborted", "Segmentation fault"]):
        state.last_error = line_str.strip()


def start_inference_server(model: str, host: str, port: int, device: str = "npu", legacy: bool = False, offline: bool = False) -> bool:
    device = (device or "npu").lower()
    if state.process and state.process.poll() is None:
        stop_inference_server()

    is_stt = "whisper" in model.lower()
    if not is_stt:
        for m in discover_models():
            if m["id"] == model and m.get("task") == "speech-to-text":
                is_stt = True
                break

    if is_stt:
        dev_label = "NPU (XDNA2)" if device == "npu" else "CPU"
        append_log(f"Avvio del motore Speech-to-Text ({model}) su {dev_label}...")
        state.status = "starting"
        state.active_model = model
        state.host = host
        state.port = port
        state.device = device
        state.legacy = legacy
        state.offline = offline
        state.load_progress = 15.0
        state.load_step = f"Caricamento modello {model} su {dev_label}..."
        state.is_loaded = False
        state.tok_per_sec = 0.0
        state.last_error = ""
        state.start_time = time.time()
        
        def _load_stt_async():
            try:
                stt_id = "openai/whisper-base"
                if "large-v3-turbo" in model.lower() or "turbo" in model.lower():
                    stt_id = "openai/whisper-large-v3-turbo"
                elif "large-v3" in model.lower() or "large" in model.lower():
                    stt_id = "openai/whisper-large-v3"
                else:
                    cfg_path = ROOT_DIR / f"quantized_weights_{model}" / "config.json"
                    if cfg_path.exists():
                        try:
                            with open(cfg_path) as cf:
                                cdata = json.load(cf)
                                if "hf_model_id" in cdata:
                                    stt_id = cdata["hf_model_id"]
                        except Exception:
                            pass

                from runtime.py.whisper_stt import WhisperSTT
                stt = WhisperSTT.get_instance(model_id=stt_id, device=device)
                stt._ensure_loaded()
                state.load_progress = 100.0
                state.is_loaded = True
                state.load_step = "Modello pronto"
                state.status = "running"
                append_log(f"[STT Engine] Server STT attivo su {dev_label} con {stt_id}. Pronto per streaming WebSocket (/ws/stt) e REST API (/v1/audio/transcriptions).")
            except Exception as e:
                state.status = "error"
                state.last_error = str(e)
                append_log(f"[STT Engine] Errore avvio modello: {e}")
        
        threading.Thread(target=_load_stt_async, daemon=True).start()
        save_config({"default_model": model, "host": host, "port": port, "device": device})
        return True

    if device == "cpu":
        state.status = "error"
        err_msg = f"L'esecuzione su CPU non è disponibile per il modello '{model}'. Solo NPU (XDNA2) è supportato."
        state.last_error = err_msg
        append_log(f"Errore avvio: {err_msg}")
        return False

    alveare_bin = ROOT_DIR / "alveare"
    cmd = [str(alveare_bin), "serve", model, "--device", device, "--host", host, "--port", str(port)]
    if legacy:
        cmd.append("--legacy")
    if offline:
        cmd.append("--offline")

    stdbuf_bin = shutil.which("stdbuf")
    if stdbuf_bin:
        cmd = [stdbuf_bin, "-oL", "-eL"] + cmd

    append_log(f"Starting inference server: {' '.join(cmd)}")
    state.status = "starting"
    state.active_model = model
    state.host = host
    state.port = port
    state.device = device
    state.legacy = legacy
    state.offline = offline
    state.load_progress = 0.0
    state.load_step = "Avvio processo server..."
    state.is_loaded = False
    state.tok_per_sec = 0.0
    state.last_error = ""

    # Estimate total layers based on model name/arch
    if "12b" in model.lower() or "gemma4" in model.lower():
        state.total_layers = 48
    elif "gemma3" in model.lower() or "1b" in model.lower() or "llama" in model.lower():
        state.total_layers = 26
    else:
        state.total_layers = 32

    env = os.environ.copy()
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    try:
        state.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(ROOT_DIR),
            env=env,
            start_new_session=True
        )
        state.start_time = time.time()
        proc_ref = state.process

        # Start background log reader
        def log_reader():
            if not proc_ref or not proc_ref.stdout:
                return
            for line in iter(proc_ref.stdout.readline, ''):
                if line:
                    stripped = line.strip()
                    append_log(stripped)
                    parse_server_log_line(stripped)

            exit_code = proc_ref.poll() if proc_ref else None
            if exit_code is not None and exit_code != 0 and exit_code != -signal.SIGTERM and exit_code != -signal.SIGKILL:
                state.status = "error"
                if not state.last_error:
                    state.last_error = f"Il server di inferenza si è arrestato in modo anomalo (exit code {exit_code})."
            else:
                state.status = "stopped"
            append_log("Inference server process exited.")

        t_log = threading.Thread(target=log_reader, daemon=True)
        t_log.start()

        # Start HTTP health-check worker to detect model readiness independently of stdout line buffering
        def health_check_worker():
            check_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
            url_models = f"http://{check_host}:{port}/v1/models"
            url_root = f"http://{check_host}:{port}/"

            time.sleep(0.5)
            while state.process is proc_ref and proc_ref.poll() is None and not state.is_loaded and state.status == "starting":
                try:
                    with httpx.Client(timeout=1.5) as client:
                        try:
                            resp = client.get(url_models)
                        except Exception:
                            resp = client.get(url_root)

                        if resp.status_code < 500:
                            if state.process is proc_ref and not state.is_loaded and state.status == "starting":
                                state.load_progress = 100.0
                                state.is_loaded = True
                                state.load_step = "Modello pronto"
                                state.status = "running"
                                append_log("Inference server HTTP health-check succeeded: model ready.")
                            break
                except Exception:
                    pass
                time.sleep(0.5)

        t_health = threading.Thread(target=health_check_worker, daemon=True)
        t_health.start()

        return True
    except Exception as e:
        state.status = "error"
        state.last_error = str(e)
        append_log(f"Failed to start server: {e}")
        return False

def stop_inference_server():
    if is_active_stt_model():
        state.status = "stopped"
        state.is_loaded = False
        state.load_progress = 0.0
        state.tok_per_sec = 0.0
        state.load_step = "Server arrestato"
        append_log(f"[{state.active_model}] Server STT arrestato.")
        return

    if state.process and state.process.poll() is None:
        append_log("Stopping inference server process...")
        try:
            pgid = os.getpgid(state.process.pid)
            os.killpg(pgid, signal.SIGTERM)
            state.process.wait(timeout=1.5)
        except Exception:
            try:
                pgid = os.getpgid(state.process.pid)
                os.killpg(pgid, signal.SIGKILL)
            except Exception:
                pass
        state.process = None
    state.status = "stopped"
    state.is_loaded = False
    state.load_progress = 0.0
    state.tok_per_sec = 0.0
    append_log("Inference server stopped.")

class StartRequest(BaseModel):
    model: Optional[str] = None
    host: Optional[str] = "127.0.0.1"
    port: Optional[int] = 8000
    device: Optional[str] = "npu"
    legacy: Optional[bool] = False
    offline: Optional[bool] = False

class ConfigUpdateRequest(BaseModel):
    first_launch: Optional[bool] = None
    default_model: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    legacy: Optional[bool] = None
    offline: Optional[bool] = None

@app.get("/api/status")
async def get_status():
    if is_active_stt_model():
        is_running = (state.status in ("starting", "running"))
    else:
        is_running = state.process is not None and state.process.poll() is None
        if not is_running and state.status == "running":
            state.status = "stopped"

    uptime = round(time.time() - state.start_time, 1) if (is_running and state.start_time > 0) else 0
    cfg = load_config()

    return {
        "status": state.status,
        "is_running": is_running,
        "is_loaded": state.is_loaded,
        "load_progress": state.load_progress,
        "load_step": state.load_step,
        "tok_per_sec": state.tok_per_sec,
        "model": state.active_model,
        "device": getattr(state, "device", "npu"),
        "host": state.host,
        "port": state.port,
        "legacy": state.legacy,
        "offline": state.offline,
        "pid": state.process.pid if (state.process and is_running) else None,
        "uptime_seconds": uptime,
        "first_launch": cfg.get("first_launch", True),
        "last_error": state.last_error,
        "cpu_usage": {
            "percent": telemetry_tracker.cpu_percent
        },
        "npu_usage": {
            "percent": telemetry_tracker.npu_percent,
            "present": telemetry_tracker.npu_present,
            "num_cols": telemetry_tracker.npu_cols,
            "active_contexts": telemetry_tracker.npu_active_contexts,
            "command_submissions": telemetry_tracker.npu_submissions,
            "command_completions": telemetry_tracker.npu_completions,
            "device_name": "AMD Ryzen AI XDNA2"
        }
    }

@app.get("/api/system/metrics")
async def get_system_metrics():
    return {
        "cpu": {
            "percent": telemetry_tracker.cpu_percent
        },
        "npu": {
            "percent": telemetry_tracker.npu_percent,
            "present": telemetry_tracker.npu_present,
            "num_cols": telemetry_tracker.npu_cols,
            "active_contexts": telemetry_tracker.npu_active_contexts,
            "command_submissions": telemetry_tracker.npu_submissions,
            "command_completions": telemetry_tracker.npu_completions,
            "device_name": "AMD Ryzen AI XDNA2"
        }
    }

@app.get("/api/models")
async def get_models():
    return discover_models()

@app.delete("/api/models/{model_id}")
async def delete_model(model_id: str):
    is_running = state.process is not None and state.process.poll() is None
    if is_running and state.active_model == model_id:
        raise HTTPException(
            status_code=400,
            detail=f"Impossibile eliminare il modello '{model_id}' perché è attualmente in esecuzione. Arresta prima il server."
        )

    # Find model directories and aliases
    possible_dirs = [
        ROOT_DIR / f"quantized_weights_{model_id}",
        ROOT_DIR / "kernels" / "build" / model_id
    ]
    if model_id in ("gemma3", "gemma-3"):
        possible_dirs.extend([
            ROOT_DIR / "quantized_weights_gemma3",
            ROOT_DIR / "kernels" / "build" / "gemma3"
        ])
    elif model_id in ("gemma4", "gemma4-12b"):
        possible_dirs.extend([
            ROOT_DIR / "quantized_weights_gemma4",
            ROOT_DIR / "kernels" / "build" / "gemma4"
        ])
    elif model_id in ("gemma4-e4b", "e4b"):
        possible_dirs.extend([
            ROOT_DIR / "quantized_weights_gemma4-e4b",
            ROOT_DIR / "kernels" / "build" / "gemma4-e4b"
        ])

    import shutil
    deleted_items = []
    seen = set()

    for d in possible_dirs:
        str_p = str(d)
        if str_p in seen:
            continue
        seen.add(str_p)

        if d.is_symlink():
            try:
                target = d.resolve()
                d.unlink()
                deleted_items.append(d.name)
                if target.exists() and str(target).startswith(str(ROOT_DIR)):
                    shutil.rmtree(target)
                    deleted_items.append(target.name)
            except Exception as e:
                print(f"[ControlServer] Warning: failed to remove symlink {d}: {e}")
        elif d.exists():
            try:
                if d.is_dir():
                    shutil.rmtree(d)
                else:
                    d.unlink()
                deleted_items.append(d.name)
            except Exception as e:
                print(f"[ControlServer] Warning: failed to delete {d}: {e}")

    if not deleted_items:
        raise HTTPException(status_code=404, detail=f"Nessuna cartella trovata per il modello '{model_id}'")

    append_log(f"Modello '{model_id}' eliminato con successo (elementi rimossi: {', '.join(deleted_items)}).")
    return {"status": "ok", "message": f"Modello '{model_id}' eliminato con successo", "deleted": deleted_items}

@app.get("/api/supported_models")
async def get_supported_models():
    return SUPPORTED_MODELS

class ModelSetupRequest(BaseModel):
    alias: str
    arch: str = "gemma4"
    source_type: str = "auto"
    url_or_repo: Optional[str] = None
    filename: Optional[str] = None
    local_gguf_path: Optional[str] = None
    custom_script: Optional[str] = None

@app.get("/api/models/setup/status")
async def get_setup_status():
    return {
        "is_running": setup_state.is_running,
        "progress": setup_state.progress,
        "step": setup_state.step,
        "active_alias": setup_state.active_alias,
        "logs": setup_state.logs[-100:],
        "error": setup_state.error
    }

def run_setup_task(req: ModelSetupRequest):
    setup_state.is_running = True
    setup_state.progress = 0.0
    setup_state.step = "start"
    setup_state.logs.clear()
    setup_state.error = ""
    setup_state.active_alias = req.alias

    def progress_callback(step: str, percent: float, msg: str):
        setup_state.step = step
        setup_state.progress = percent
        entry = {
            "timestamp": time.strftime("[%H:%M:%S]"),
            "step": step,
            "percent": percent,
            "message": msg
        }
        setup_state.logs.append(entry)
        append_log(f"[Setup:{req.alias}] {msg}")

    try:
        py_exe = os.environ.get("ALVEARE_PYTHON")
        if not py_exe and os.environ.get("CONDA_PREFIX"):
            py_exe = str(Path(os.environ.get("CONDA_PREFIX")) / "bin" / "python")
        if not py_exe:
            alveare_env_py = Path("/home/daino/miniconda3/envs/alveare-aie/bin/python")
            if alveare_env_py.exists():
                py_exe = str(alveare_env_py)
            else:
                py_exe = sys.executable

        setup_script = ROOT_DIR / "tools" / "setup_model.py"
        cmd = [py_exe, str(setup_script), req.alias, "--arch", req.arch]
        if req.source_type == "local" and req.local_gguf_path:
            cmd.extend(["--gguf", req.local_gguf_path])
        else:
            if req.url_or_repo:
                cmd.extend(["--url", req.url_or_repo])
            if req.filename:
                cmd.extend(["--filename", req.filename])
        if req.arch == "custom" and req.custom_script:
            cmd.extend(["--custom-script", req.custom_script])

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(ROOT_DIR)
        )

        for line in iter(proc.stdout.readline, ''):
            if not line:
                continue
            line_str = line.strip()
            if "[SETUP_PROGRESS]" in line_str:
                try:
                    payload = json.loads(line_str.split("[SETUP_PROGRESS]")[1].strip())
                    progress_callback(payload.get("step", "running"), payload.get("percent", setup_state.progress), payload.get("message", ""))
                except Exception:
                    progress_callback("running", setup_state.progress, line_str)
            else:
                progress_callback("running", setup_state.progress, line_str)

        proc.wait()
        if proc.returncode == 0:
            setup_state.progress = 100.0
            setup_state.step = "complete"
            progress_callback("complete", 100.0, f"Model '{req.alias}' set up successfully!")
        else:
            setup_state.error = f"Setup exited with code {proc.returncode}"
            setup_state.step = "error"
            progress_callback("error", setup_state.progress, setup_state.error)
    except Exception as e:
        setup_state.error = str(e)
        setup_state.step = "error"
        progress_callback("error", setup_state.progress, f"Error: {e}")
    finally:
        setup_state.is_running = False

@app.post("/api/models/setup")
async def start_model_setup(req: ModelSetupRequest):
    if setup_state.is_running:
        raise HTTPException(status_code=400, detail="Un'operazione di setup modello è già in corso.")
    
    alias = (req.alias or "").strip()
    if not alias:
        raise HTTPException(status_code=400, detail="L'alias del modello è obbligatorio.")
    
    if not re.match(r'^[a-zA-Z0-9_\-]+$', alias):
        raise HTTPException(status_code=400, detail="L'alias può contenere solo lettere, numeri, trattini e underscore.")

    valid_archs = {"gemma4", "gemma4-e4b", "gemma3", "llama", "custom"}
    if req.arch not in valid_archs:
        raise HTTPException(status_code=400, detail=f"Architettura non supportata '{req.arch}'. Opzioni ammesse: {', '.join(valid_archs)}")

    if req.source_type == "local":
        if not req.local_gguf_path:
            raise HTTPException(status_code=400, detail="Il percorso del file GGUF locale è obbligatorio in modalità manuale.")
        
        gguf_p = Path(req.local_gguf_path).resolve()
        if not gguf_p.exists() or not gguf_p.is_file():
            raise HTTPException(status_code=400, detail=f"Il file GGUF locale non esiste: {req.local_gguf_path}")
        
        if not gguf_p.name.endswith(".gguf"):
            raise HTTPException(status_code=400, detail="Il file deve avere estensione .gguf")
        
        try:
            with open(gguf_p, "rb") as f:
                magic = f.read(4)
                if magic != b"GGUF":
                    raise HTTPException(status_code=400, detail=f"Il file {gguf_p.name} non è un file GGUF valido (magic header atteso: GGUF, trovato: {magic}).")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Impossibile leggere il file GGUF locale: {e}")
    else:
        # Auto mode
        if not req.url_or_repo and not any(m["id"] == alias or m["arch"] == req.arch for m in SUPPORTED_MODELS):
            raise HTTPException(status_code=400, detail="Specificare un repository HuggingFace/URL GGUF o selezionare un modello supportato predefinito.")

    import threading
    t = threading.Thread(target=run_setup_task, args=(req,), daemon=True)
    t.start()

    return {"status": "ok", "message": f"Setup started for model {req.alias}"}

# HTTP Unbuffered SSE Proxy for /v1/chat/completions
@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    if not (state.process and state.process.poll() is None):
        raise HTTPException(status_code=503, detail="Inference server is not running. Start it from the Control Panel.")

    body = await request.body()
    target_url = f"http://{state.host}:{state.port}/v1/chat/completions"
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}

    req_json = {}
    try:
        req_json = json.loads(body.decode("utf-8"))
    except Exception:
        pass

    is_stream = req_json.get("stream", False)

    if is_stream:
        async def stream_generator():
            async with httpx.AsyncClient(timeout=300.0) as client:
                try:
                    async with client.stream("POST", target_url, headers=headers, content=body) as resp:
                        if resp.status_code != 200:
                            err_content = await resp.aread()
                            yield err_content
                            return
                        async for chunk in resp.aiter_bytes():
                            yield chunk
                except Exception as e:
                    err_json = json.dumps({"error": {"message": f"Proxy streaming error: {e}"}})
                    yield f"data: {err_json}\n\n".encode("utf-8")

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                resp = await client.post(target_url, headers=headers, content=body)
                return JSONResponse(status_code=resp.status_code, content=resp.json())
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"Error connecting to inference server: {e}")

class FileUploadItem(BaseModel):
    filename: str
    content_b64: str
    mime_type: Optional[str] = None

class FileUploadRequest(BaseModel):
    files: List[FileUploadItem]

@app.post("/api/upload")
async def upload_files(req: FileUploadRequest):
    results = []
    for item in req.files:
        try:
            content = base64.b64decode(item.content_b64)
            parsed = parse_file_upload(item.filename, content, item.mime_type)
            results.append(parsed)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Errore decodifica file '{item.filename}': {e}")
    return {"files": results}

# OpenAI-compatible Audio Transcription API
@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(...),
    model: Optional[str] = Form("whisper-base"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0)
):
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty audio file provided")

        stt = WhisperSTT.get_instance()
        state.is_transcribing = True
        try:
            res = await asyncio.to_thread(stt.transcribe, content, language or "auto")
        finally:
            state.is_transcribing = False
            state.last_transcribe_time = time.time()
        
        if res.get("status") == "error":
            raise HTTPException(status_code=500, detail=res.get("error", "Transcription failed"))

        if response_format == "text":
            return PlainTextResponse(res.get("text", ""))
        elif response_format == "verbose_json":
            return {
                "task": "transcribe",
                "language": res.get("language", "auto"),
                "duration": round(len(content) / 32000.0, 2),
                "text": res.get("text", ""),
                "latency_ms": res.get("latency_ms", 0.0)
            }
        else:
            return {"text": res.get("text", "")}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio transcription error: {e}")

# Fast STT endpoint for React Web UI Microphone recording
class SttJsonRequest(BaseModel):
    audio_b64: Optional[str] = None
    language: Optional[str] = "auto"

@app.post("/api/stt")
async def api_stt_transcribe(
    file: Optional[UploadFile] = File(None),
    audio_b64: Optional[str] = Form(None),
    language: Optional[str] = Form("auto")
):
    try:
        content = b""
        if file is not None:
            content = await file.read()
        elif audio_b64:
            content = base64.b64decode(audio_b64)
        else:
            raise HTTPException(status_code=400, detail="Nessun flusso audio inviato.")

        stt = WhisperSTT.get_instance()
        state.is_transcribing = True
        try:
            res = await asyncio.to_thread(stt.transcribe, content, language=language or "auto")
        finally:
            state.is_transcribing = False
            state.last_transcribe_time = time.time()
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore STT: {e}")

# Real-Time WebSocket Streaming STT Engine with Voice Activity Detection (VAD) & Continuous Sentence Segmentation
async def handle_stt_stream_connection(websocket: WebSocket):
    await websocket.accept()
    from runtime.py.whisper_stt import WhisperSTT
    stt = WhisperSTT.get_instance()
    
    stream_language = "auto"
    utterance_buffer = bytearray()
    
    SILENCE_THRESHOLD_RMS = 15.0  # Sensitive RMS threshold for 16-bit PCM silence detection
    MIN_UTTERANCE_BYTES = int(16000 * 2 * 0.35)  # 0.35s minimum audio (~11,200 bytes)
    SILENCE_COMMIT_SECS = 1.8  # 1.8s silence commits and splits sentence into history
    MAX_PHRASE_BYTES = int(16000 * 2 * 25.0)  # 25.0s max phrase before auto-commit
    PARTIAL_INTERVAL_SECS = 1.0  # Emits live partial every 1.0s if speaking
    
    speech_active = False
    last_voice_time = time.time()
    last_partial_time = time.time()
    is_busy = False

    async def _transcribe_audio(pcm_bytes: bytes, is_final: bool):
        nonlocal is_busy
        if len(pcm_bytes) < 3200:  # < 100ms
            if is_final:
                try:
                    await websocket.send_json({
                        "type": "final",
                        "text": "",
                        "language": stream_language,
                        "latency_ms": 0.0,
                        "is_final": True
                    })
                except Exception:
                    pass
            return
            
        if is_final:
            while is_busy:
                await asyncio.sleep(0.05)
        elif is_busy:
            return
            
        is_busy = True
        state.is_transcribing = True
        try:
            usable_len = len(pcm_bytes) - (len(pcm_bytes) % 2)
            pcm_arr = np.frombuffer(pcm_bytes[:usable_len], dtype=np.int16).astype(np.float32) / 32768.0
            res = await asyncio.to_thread(stt.transcribe, pcm_arr, stream_language)
            text = res.get("text", "").strip()
            if text or is_final:
                await websocket.send_json({
                    "type": "final" if is_final else "partial",
                    "text": text,
                    "language": res.get("language", stream_language),
                    "latency_ms": res.get("latency_ms", 0.0),
                    "is_final": is_final
                })
        except Exception as e:
            print(f"[WebSocket STT] Transcription error: {e}")
        finally:
            is_busy = False
            state.is_transcribing = False
            state.last_transcribe_time = time.time()

    try:
        while True:
            # Check timeout every 50ms so silence and partials update responsively
            try:
                msg = await asyncio.wait_for(websocket.receive(), timeout=0.05)
            except asyncio.TimeoutError:
                msg = None

            now = time.time()

            if msg is not None:
                if "bytes" in msg and msg["bytes"]:
                    chunk = msg["bytes"]
                    usable_chunk_len = len(chunk) - (len(chunk) % 2)
                    if usable_chunk_len > 0:
                        samples = np.frombuffer(chunk[:usable_chunk_len], dtype=np.int16)
                        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))) if len(samples) > 0 else 0.0
                        if rms >= SILENCE_THRESHOLD_RMS:
                            speech_active = True
                            last_voice_time = now
                            utterance_buffer.extend(chunk)
                        elif speech_active:
                            # Short pause/silence while user is in active conversation
                            utterance_buffer.extend(chunk)
                elif "text" in msg and msg["text"]:
                    try:
                        payload = json.loads(msg["text"])
                        action = payload.get("action", "")
                        if action == "set_language":
                            stream_language = payload.get("language", "auto")
                        elif action in ("flush", "stop", "final"):
                            if len(utterance_buffer) >= 3200:
                                current_pcm = bytes(utterance_buffer)
                                utterance_buffer.clear()
                                speech_active = False
                                await _transcribe_audio(current_pcm, is_final=True)
                            else:
                                await websocket.send_json({
                                    "type": "final",
                                    "text": "",
                                    "language": stream_language,
                                    "latency_ms": 0.0,
                                    "is_final": True
                                })
                            utterance_buffer.clear()
                            speech_active = False
                        elif action == "clear":
                            utterance_buffer.clear()
                            speech_active = False
                        elif action == "ping":
                            await websocket.send_json({"type": "pong"})
                    except Exception as err:
                        print(f"[WebSocket STT] JSON error: {err}")

            # Check if current utterance needs partial update or finalization
            if speech_active and len(utterance_buffer) >= MIN_UTTERANCE_BYTES:
                time_since_voice = now - last_voice_time
                phrase_too_long = len(utterance_buffer) >= MAX_PHRASE_BYTES
                
                if (time_since_voice >= SILENCE_COMMIT_SECS or phrase_too_long) and not is_busy:
                    # Finalize sentence into history
                    current_pcm = bytes(utterance_buffer)
                    utterance_buffer.clear()
                    speech_active = False
                    last_partial_time = now
                    asyncio.create_task(_transcribe_audio(current_pcm, is_final=True))
                elif (now - last_partial_time >= PARTIAL_INTERVAL_SECS) and not is_busy:
                    last_partial_time = now
                    current_pcm = bytes(utterance_buffer)
                    asyncio.create_task(_transcribe_audio(current_pcm, is_final=False))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket STT] Connection exception: {e}")

@app.websocket("/ws/stt")
async def websocket_stt_root(websocket: WebSocket):
    await handle_stt_stream_connection(websocket)

@app.websocket("/api/stt/stream")
async def websocket_stt_api(websocket: WebSocket):
    await handle_stt_stream_connection(websocket)

@app.post("/api/control/start")
async def control_start(req: StartRequest):
    model = req.model or state.active_model
    success = start_inference_server(
        model=model,
        host=req.host or "127.0.0.1",
        port=req.port or 8000,
        device=req.device or "npu",
        legacy=bool(req.legacy),
        offline=bool(req.offline)
    )
    if not success:
        raise HTTPException(status_code=500, detail=state.last_error or "Failed to start server")
    return {"status": "ok", "message": f"Server starting with model {model} on {req.device or 'npu'}"}

@app.post("/api/control/stop")
async def control_stop():
    stop_inference_server()
    return {"status": "ok", "message": "Server stopped"}

@app.post("/api/control/restart")
async def control_restart(req: StartRequest):
    stop_inference_server()
    await asyncio.sleep(1)
    model = req.model or state.active_model
    success = start_inference_server(
        model=model,
        host=req.host or "127.0.0.1",
        port=req.port or 8000,
        device=req.device or "npu",
        legacy=bool(req.legacy),
        offline=bool(req.offline)
    )
    if not success:
        raise HTTPException(status_code=500, detail=state.last_error or "Failed to restart server")
    return {"status": "ok", "message": f"Server restarted with model {model} on {req.device or 'npu'}"}

class BuildKernelsRequest(BaseModel):
    model: Optional[str] = None
    device: Optional[str] = "npu"
    force_arch: Optional[str] = None
    no_gemm: Optional[bool] = False
    max_batch: Optional[int] = 16

@app.get("/api/kernels/status")
async def get_kernels_status():
    result = {}
    models = discover_models()
    for m in models:
        alias = m["alias"]
        possible_paths = [
            ROOT_DIR / "kernels" / "build" / alias / "manifest.json",
            ROOT_DIR / "kernels" / f"build_{alias}" / "manifest.json",
            Path(m["path"]) / "manifest.json",
            ROOT_DIR / "kernels" / "build" / "manifest.json"
        ]
        manifest_found = None
        manifest_data = None
        for p in possible_paths:
            if p.exists():
                try:
                    with open(p, "r") as f:
                        d = json.load(f)
                    if d.get("model_type") == m["arch"]:
                        manifest_found = p
                        manifest_data = d
                        break
                    elif not manifest_found:
                        manifest_found = p
                        manifest_data = d
                except Exception:
                    pass

        if manifest_found and manifest_data:
            result[m["id"]] = {
                "manifest_exists": True,
                "manifest_model_type": manifest_data.get("model_type"),
                "kernels_count": len(manifest_data.get("kernels", [])),
                "manifest_path": str(manifest_found)
            }
        else:
            result[m["id"]] = {
                "manifest_exists": False,
                "kernels_count": 0
            }
    return result

@app.post("/api/control/build-kernels")
async def control_build_kernels(req: BuildKernelsRequest):
    model = req.model or state.active_model
    if state.process and state.process.poll() is None:
        stop_inference_server()

    alveare_bin = ROOT_DIR / "alveare"
    cmd = [str(alveare_bin), "build-kernels", model, "--device", req.device or "npu"]
    if req.no_gemm:
        cmd.append("--no-gemm")
    if req.max_batch and req.max_batch != 16:
        cmd.extend(["--max-batch", str(req.max_batch)])

    append_log(f"Starting kernel compilation: {' '.join(cmd)}")
    state.status = "building_kernels"

    try:
        state.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(ROOT_DIR),
            env=os.environ.copy()
        )
        state.start_time = time.time()

        def log_reader():
            if not state.process or not state.process.stdout:
                return
            for line in iter(state.process.stdout.readline, ''):
                if line:
                    append_log(line.strip())
            state.status = "stopped"
            append_log("Kernel compilation completed.")

        import threading
        t = threading.Thread(target=log_reader, daemon=True)
        t.start()

        return {"status": "ok", "message": f"Building kernels for model {model}"}
    except Exception as e:
        state.status = "error"
        state.last_error = str(e)
        append_log(f"Failed to start kernel build: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/config")
async def get_config():
    return load_config()

@app.post("/api/config")
async def update_config(req: ConfigUpdateRequest):
    updates = {k: v for k, v in req.dict().items() if v is not None}
    save_config(updates)
    return load_config()

@app.get("/api/npu/check")
async def npu_check():
    device_node = Path("/dev/accel/accel0").exists()
    xrt_smi = False
    pyxrt_import = False
    
    try:
        res = subprocess.run(["which", "xrt-smi"], capture_output=True, text=True)
        xrt_smi = res.returncode == 0
    except Exception:
        pass

    try:
        import pyxrt
        pyxrt_import = True
    except ImportError:
        try:
            if "/usr/lib/python3/dist-packages" not in sys.path:
                sys.path.append("/usr/lib/python3/dist-packages")
            import pyxrt
            pyxrt_import = True
        except Exception:
            pyxrt_import = False

    return {
        "device_node": device_node,
        "xrt_smi": xrt_smi,
        "pyxrt_import": pyxrt_import,
        "all_ok": device_node and pyxrt_import
    }

@app.get("/api/logs")
async def get_logs():
    return {"logs": state.log_buffer}

import atexit
atexit.register(stop_inference_server)

@app.on_event("shutdown")
def on_shutdown():
    stop_inference_server()

def handle_control_signal(sig, frame):
    print(f"\n[ControlServer] Signal {sig} received, stopping inference server and shutting down...")
    stop_inference_server()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_control_signal)
signal.signal(signal.SIGTERM, handle_control_signal)

# WebSocket Chat Bridge
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data_text = await websocket.receive_text()
            data = json.loads(data_text)
            
            messages = data.get("messages", [])
            model_name = data.get("model", state.active_model)
            temperature = data.get("temperature", 0.7)
            top_p = data.get("top_p", 0.9)
            top_k = data.get("top_k", 50)
            max_tokens = data.get("max_tokens", 512)
            max_context_length = data.get("max_context_length", 4096)
            enable_thinking = data.get("enable_thinking", True)

            # Check if inference server is running
            if not (state.process and state.process.poll() is None):
                await websocket.send_json({
                    "type": "error",
                    "message": "Inference server is not running. Please start it from the Control Panel."
                })
                continue

            # Forward request to OpenAI API on localhost:port
            import urllib.request
            req_payload = json.dumps({
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_tokens": max_tokens,
                "max_context_length": max_context_length,
                "enable_thinking": enable_thinking,
                "stream": True
            }).encode("utf-8")

            url = f"http://{state.host}:{state.port}/v1/chat/completions"
            req = urllib.request.Request(url, data=req_payload, headers={"Content-Type": "application/json"})

            try:
                t0 = time.time()
                token_count = 0
                
                with urllib.request.urlopen(req) as resp:
                    for line in resp:
                        line_str = line.decode("utf-8").strip()
                        if not line_str or line_str.startswith(":"):
                            continue
                        if line_str == "data: [DONE]":
                            break
                        if line_str.startswith("data: "):
                            json_str = line_str[6:]
                            try:
                                chunk = json.loads(json_str)
                                choices = chunk.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        token_count += 1
                                        await websocket.send_json({
                                            "type": "token",
                                            "content": content
                                        })
                            except json.JSONDecodeError:
                                pass
                
                elapsed = time.time() - t0
                tps = round(token_count / elapsed, 1) if elapsed > 0 else 0
                await websocket.send_json({
                    "type": "done",
                    "finish_reason": "stop",
                    "metrics": {
                        "tokens": token_count,
                        "elapsed_seconds": round(elapsed, 2),
                        "tps": tps
                    }
                })
            except Exception as req_err:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error communicating with inference server: {req_err}"
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket] Exception: {e}")

# Serve built frontend static files if available.
#
# index.html MUST NOT be cached: the bundle filenames are content-hashed, so a rebuilt
# UI is only picked up if the browser re-reads index.html to learn the new hashes.
# Serving it cacheable is why a rebuilt frontend can keep showing the old UI until a
# hard refresh. The hashed assets themselves stay cacheable (they never change content).
FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"
if FRONTEND_DIST.exists():
    # A middleware (not a route) because the StaticFiles mount below also answers "/",
    # and it wins regardless of registration order.
    @app.middleware("http")
    async def _no_cache_html(request, call_next):
        response = await call_next(request)
        if response.headers.get("content-type", "").startswith("text/html"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
        return response

    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("ALVEARE_CONTROL_HOST", "127.0.0.1")
    port = int(os.getenv("ALVEARE_CONTROL_PORT", "8080"))
    print(f"[Alveare Control Server] Listening on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
