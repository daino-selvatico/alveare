import os
import sys
import time
import json
import asyncio
import subprocess
import signal
import re
import httpx
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Root directory of the repository
ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_FILE = ROOT_DIR / ".alveare_config.json"

app = FastAPI(title="Alveare Control & Web UI Backend")

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

class SetupState:
    def __init__(self):
        self.is_running: bool = False
        self.progress: float = 0.0
        self.step: str = ""
        self.logs: List[Dict[str, Any]] = []
        self.error: str = ""
        self.active_alias: str = ""

state = ServerState()
setup_state = SetupState()

def load_config() -> Dict[str, Any]:
    default_config = {
        "first_launch": True,
        "default_model": "gemma4",
        "host": "127.0.0.1",
        "port": 8000,
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
        arch = "unknown"
        if cfg_path.exists():
            try:
                with open(cfg_path, "r") as f:
                    cdata = json.load(f)
                    arch = cdata.get("model_type", "unknown")
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

        models.append({
            "id": alias,
            "alias": alias,
            "arch": arch,
            "path": str(real_p),
            "size_mb": size_mb,
            "has_config": cfg_path.exists(),
            "active": (alias == state.active_model)
        })
    return models

# Regex patterns for parsing C++ / Python server stdout
LAYER_LOAD_RE = re.compile(r'(?:Loading weights for layer|Loading layer|Pre-packing FFN fused weights for layer|layer)\s+(\d+)(?:/(\d+))?', re.IGNORECASE)
TOKEN_SPEED_RE = re.compile(r'(?:Token\s+\d+/\d+\s+in|generated in)\s+([\d\.]+)\s*ms', re.IGNORECASE)
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
    m_token = TOKEN_SPEED_RE.search(line_str)
    if m_token:
        try:
            ms = float(m_token.group(1))
            if ms > 0:
                instant_tps = 1000.0 / ms
                state.tok_per_sec = round(instant_tps if state.tok_per_sec == 0 else (state.tok_per_sec * 0.7 + instant_tps * 0.3), 1)
        except Exception:
            pass

    m_tps = TPS_RE.search(line_str)
    if m_tps:
        try:
            state.tok_per_sec = round(float(m_tps.group(1)), 1)
        except Exception:
            pass

    # 3. Error detection
    if any(err_kw in line_str for err_kw in ["Error:", "Exception:", "CUDA error", "XRT error", "FATAL", "Aborted", "Segmentation fault"]):
        state.last_error = line_str.strip()


def start_inference_server(model: str, host: str, port: int, legacy: bool, offline: bool) -> bool:
    if state.process and state.process.poll() is None:
        stop_inference_server()

    alveare_bin = ROOT_DIR / "alveare"
    cmd = [str(alveare_bin), "serve", model, "--host", host, "--port", str(port)]
    if legacy:
        cmd.append("--legacy")
    if offline:
        cmd.append("--offline")

    append_log(f"Starting inference server: {' '.join(cmd)}")
    state.status = "starting"
    state.active_model = model
    state.host = host
    state.port = port
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
        
        # Start background log reader
        def log_reader():
            if not state.process or not state.process.stdout:
                return
            for line in iter(state.process.stdout.readline, ''):
                if line:
                    stripped = line.strip()
                    append_log(stripped)
                    parse_server_log_line(stripped)
            
            exit_code = state.process.poll() if state.process else None
            if exit_code is not None and exit_code != 0 and exit_code != -signal.SIGTERM and exit_code != -signal.SIGKILL:
                state.status = "error"
                if not state.last_error:
                    state.last_error = f"Il server di inferenza si è arrestato in modo anomalo (exit code {exit_code})."
            else:
                state.status = "stopped"
            append_log("Inference server process exited.")

        import threading
        t = threading.Thread(target=log_reader, daemon=True)
        t.start()

        return True
    except Exception as e:
        state.status = "error"
        state.last_error = str(e)
        append_log(f"Failed to start server: {e}")
        return False

def stop_inference_server():
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
    is_running = state.process is not None and state.process.poll() is None
    if not is_running and state.status == "running":
        state.status = "stopped"

    uptime = round(time.time() - state.start_time, 1) if is_running else 0
    cfg = load_config()

    return {
        "status": state.status,
        "is_running": is_running,
        "is_loaded": state.is_loaded,
        "load_progress": state.load_progress,
        "load_step": state.load_step,
        "tok_per_sec": state.tok_per_sec,
        "model": state.active_model,
        "host": state.host,
        "port": state.port,
        "legacy": state.legacy,
        "offline": state.offline,
        "pid": state.process.pid if is_running else None,
        "uptime_seconds": uptime,
        "first_launch": cfg.get("first_launch", True),
        "last_error": state.last_error
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

@app.post("/api/control/start")
async def control_start(req: StartRequest):
    model = req.model or state.active_model
    success = start_inference_server(
        model=model,
        host=req.host or "127.0.0.1",
        port=req.port or 8000,
        legacy=bool(req.legacy),
        offline=bool(req.offline)
    )
    if not success:
        raise HTTPException(status_code=500, detail=state.last_error or "Failed to start server")
    return {"status": "ok", "message": f"Server starting with model {model}"}

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
        legacy=bool(req.legacy),
        offline=bool(req.offline)
    )
    if not success:
        raise HTTPException(status_code=500, detail=state.last_error or "Failed to restart server")
    return {"status": "ok", "message": f"Server restarted with model {model}"}

class BuildKernelsRequest(BaseModel):
    model: Optional[str] = None
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
    cmd = [str(alveare_bin), "build-kernels", model]
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

# Serve built frontend static files if available
FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("ALVEARE_CONTROL_HOST", "127.0.0.1")
    port = int(os.getenv("ALVEARE_CONTROL_PORT", "8080"))
    print(f"[Alveare Control Server] Listening on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
