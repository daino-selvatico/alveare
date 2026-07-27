import os
import sys
import time
import json
import asyncio
import subprocess
import signal
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

state = ServerState()

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
    # Search for quantized_weights* directories and symlinks
    for d in sorted(ROOT_DIR.glob("quantized_weights*")):
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
        if folder_name.startswith("quantized_weights_"):
            alias = folder_name[len("quantized_weights_"):]
        elif folder_name == "quantized_weights":
            alias = "gemma3"
        else:
            alias = folder_name

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
            env=env
        )
        state.start_time = time.time()
        
        # Start background log reader
        def log_reader():
            if not state.process or not state.process.stdout:
                return
            for line in iter(state.process.stdout.readline, ''):
                if line:
                    append_log(line.strip())
            state.status = "stopped"
            append_log("Inference server process exited.")

        import threading
        t = threading.Thread(target=log_reader, daemon=True)
        t.start()

        state.status = "running"
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
            state.process.terminate()
            state.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            state.process.kill()
        state.process = None
    state.status = "stopped"
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
