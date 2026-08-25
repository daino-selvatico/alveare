"""
Model Downloader & Automatic Setup Pipeline for Alveare NPU.

Automates downloading GGUF models from HuggingFace, quantizing/packing weights to Q4_0,
and building NPU hardware kernels in one unified command or API service.
"""
import argparse
import json
import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from tools.convert.base_quantizer import load_quantizer_plugin

SUPPORTED_MODELS = [
    {
        "id": "gemma4",
        "name": "Gemma 4 12B Instruct",
        "arch": "gemma4",
        "default_url": "https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/resolve/main/gemma-4-12b-it-Q4_K_M.gguf",
        "repo_id": "unsloth/gemma-4-12b-it-GGUF",
        "filename": "gemma-4-12b-it-Q4_K_M.gguf",
        "size_approx": "~8.2 GB",
        "description": "State-of-the-art 12B model with 128k context window, SWA attention & high efficiency."
    },
    {
        "id": "gemma4-e4b",
        "name": "Gemma 4 E4B (Edge 4B)",
        "arch": "gemma4-e4b",
        "default_url": "https://huggingface.co/unsloth/gemma-4-e4b-it-GGUF/resolve/main/gemma-4-e4b-it-Q4_K_M.gguf",
        "repo_id": "unsloth/gemma-4-e4b-it-GGUF",
        "filename": "gemma-4-e4b-it-Q4_K_M.gguf",
        "size_approx": "~2.8 GB",
        "description": "Ultra-fast edge model designed for real-time mobile & desktop NPU inference."
    },
    {
        "id": "gemma3",
        "name": "Gemma 3 1B Instruct",
        "arch": "gemma3",
        "default_url": "https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf",
        "repo_id": "unsloth/gemma-3-1b-it-GGUF",
        "filename": "gemma-3-1b-it-Q4_K_M.gguf",
        "size_approx": "~0.8 GB",
        "description": "Lightweight 1B parameter model optimized for instant response times."
    },
    {
        "id": "llama",
        "name": "Llama 3.2 1B Instruct",
        "arch": "llama",
        "default_url": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "repo_id": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "size_approx": "~0.8 GB",
    {
        "id": "whisper-base",
        "name": "Whisper Base STT (Speech-to-Text)",
        "arch": "whisper",
        "default_url": "openai/whisper-base",
        "repo_id": "openai/whisper-base",
        "filename": "model.safetensors",
        "size_approx": "~145 MB",
        "description": "High-accuracy multilingual speech-to-text with support for Italian, English, and 90+ languages."
    }
]


def log_progress(step: str, percent: float, message: str):
    """Print structured progress log line for frontend WebSocket / log reader."""
    timestamp = time.strftime("[%H:%M:%S]")
    data = {
        "timestamp": timestamp,
        "step": step,
        "percent": round(percent, 1),
        "message": message
    }
    print(f"[SETUP_PROGRESS] {json.dumps(data)}", flush=True)


def download_gguf(url_or_repo: str, filename: Optional[str], dest_dir: Path) -> Path:
    """Download GGUF file from HuggingFace or direct URL."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 1. Try huggingface_hub if available
    try:
        from huggingface_hub import hf_hub_download
        if "/" in url_or_repo and not url_or_repo.startswith("http"):
            repo_id = url_or_repo
            file_to_dl = filename or "model.gguf"
            log_progress("download", 10.0, f"Downloading {file_to_dl} from HuggingFace repo {repo_id}...")
            local_path = hf_hub_download(repo_id=repo_id, filename=file_to_dl, local_dir=str(dest_dir))
            log_progress("download", 100.0, "Download completed via huggingface_hub.")
            return Path(local_path)
    except Exception as e:
        log_progress("download", 15.0, f"huggingface_hub unavailable or failed ({e}), falling back to direct download...")

    # 2. Direct HTTP URL download
    url = url_or_repo
    if not url.startswith("http"):
        # Convert repo_id + filename to HF URL
        fname = filename or "model.gguf"
        url = f"https://huggingface.co/{url_or_repo}/resolve/main/{fname}"

    target_name = filename or Path(url).name
    if not target_name.endswith(".gguf"):
        target_name += ".gguf"
    out_file = dest_dir / target_name

    # Remove stale or corrupted download file if smaller than 10KB
    if out_file.exists() and out_file.stat().st_size < 10000:
        out_file.unlink()

    log_progress("download", 20.0, f"Downloading from URL: {url} -> {out_file.name}")

    # Use curl with -fL flags to fail cleanly on HTTP errors
    curl_cmd = ["curl", "-fL", "-C", "-", "-o", str(out_file), url]
    res = subprocess.run(curl_cmd)
    if res.returncode != 0 or not out_file.exists():
        if out_file.exists():
            out_file.unlink()
        raise RuntimeError(f"Failed to download GGUF from {url}. Check that the URL is public and valid.")

    # Validate GGUF header magic bytes
    with open(out_file, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            out_file.unlink()
            raise ValueError(f"Downloaded file {out_file.name} is not a valid GGUF file (magic header: {magic}). The URL may require authentication or be invalid.")

    log_progress("download", 100.0, f"Download completed: {out_file.name} ({round(out_file.stat().st_size / (1024*1024), 1)} MB)")
    return out_file


def run_setup(
    alias: str,
    arch_or_quantizer: str,
    source_type: str = "auto",  # 'auto' or 'local'
    url_or_repo: Optional[str] = None,
    filename: Optional[str] = None,
    local_gguf_path: Optional[str] = None,
    custom_script: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute the end-to-end setup pipeline:
    Download (if auto) -> Quantize & Pack -> Build NPU Kernels.
    """
    log_progress("start", 0.0, f"Starting setup for model '{alias}' (Arch/Quantizer: {arch_or_quantizer})...")

    models_dir = ROOT_DIR / "models_cache"
    if arch_or_quantizer in ("whisper", "whisper-base") or alias in ("whisper", "whisper-base"):
        log_progress("download", 30.0, "Downloading and validating Whisper Base speech-to-text model...")
        from runtime.py.whisper_stt import WhisperSTT
        stt = WhisperSTT.get_instance(model_id="openai/whisper-base")
        stt._ensure_loaded()
        log_progress("complete", 100.0, "Whisper Base STT successfully initialized and ready!")
        return {
            "status": "success",
            "alias": alias,
            "weights_dir": str(models_dir / "whisper-base"),
            "kernels_dir": "",
            "config": {"model_type": "whisper", "task": "speech-to-text"}
        }

    gguf_path: Optional[Path] = None

    if source_type == "auto" or (url_or_repo and not local_gguf_path):
        if not url_or_repo:
            # Look up supported model default
            matched = next((m for m in SUPPORTED_MODELS if m["id"] == alias or m["arch"] == arch_or_quantizer), None)
            if matched:
                url_or_repo = matched["default_url"]
                filename = matched["filename"]
            else:
                raise ValueError(f"No URL or repository specified for automatic setup of '{alias}'")

        gguf_path = download_gguf(url_or_repo, filename, models_dir / alias)
    else:
        if not local_gguf_path:
            raise ValueError("Local GGUF path is required for manual mode.")
        gguf_path = Path(local_gguf_path).resolve()
        if not gguf_path.exists():
            raise FileNotFoundError(f"Local GGUF file not found: {gguf_path}")

    # 2. Quantize & Pack
    log_progress("quantize", 40.0, f"Quantizing weights from {gguf_path.name} to Q4_0 layout...")
    
    if alias in ("gemma3", "gemma"):
        weights_dir = ROOT_DIR / "quantized_weights_gemma3"
    else:
        weights_dir = ROOT_DIR / f"quantized_weights_{alias}"

    weights_dir.mkdir(parents=True, exist_ok=True)

    quantizer_spec = custom_script if arch_or_quantizer == "custom" and custom_script else arch_or_quantizer
    log_progress("quantize", 50.0, f"Loading quantizer plugin: {quantizer_spec}...")

    plugin = load_quantizer_plugin(quantizer_spec)
    if hasattr(plugin, "quantize"):
        config = plugin.quantize(str(gguf_path), str(weights_dir))
    elif hasattr(plugin, "main"):
        config = plugin.main(gguf_path=str(gguf_path), out_dir=str(weights_dir))
    else:
        raise AttributeError(f"Quantizer plugin {quantizer_spec} has no quantize() or main() method.")

    log_progress("quantize", 75.0, "Quantization and weight packing completed.")

    # 3. Build Kernels
    log_progress("build_kernels", 80.0, "Compiling / harvesting NPU AOT hardware kernels...")
    kernels_out_dir = ROOT_DIR / "kernels" / "build" / alias
    
    build_script = ROOT_DIR / "tools" / "build_kernels.py"
    py_exe = os.environ.get("ALVEARE_PYTHON")
    if not py_exe and os.environ.get("CONDA_PREFIX"):
        py_exe = str(Path(os.environ.get("CONDA_PREFIX")) / "bin" / "python")
    if not py_exe:
        alveare_env_py = Path("/home/daino/miniconda3/envs/alveare-aie/bin/python")
        if alveare_env_py.exists():
            py_exe = str(alveare_env_py)
        else:
            py_exe = sys.executable

    cmd = [py_exe, str(build_script), "--weights-dir", str(weights_dir), "--out", str(kernels_out_dir)]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        log_progress("error", 85.0, f"Kernel compilation error: {res.stderr}")
        raise RuntimeError(f"build_kernels.py failed: {res.stderr}")

    log_progress("complete", 100.0, f"Model '{alias}' successfully set up and ready to serve!")

    return {
        "status": "success",
        "alias": alias,
        "weights_dir": str(weights_dir),
        "kernels_dir": str(kernels_out_dir),
        "config": config
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alveare Model Setup Pipeline")
    parser.add_argument("alias", help="Alias for the model (e.g. gemma4, gemma3, whisper-base, mymodel)")
    parser.add_argument("--arch", default=None, help="Architecture or quantizer spec (gemma4, gemma4-e4b, gemma3, llama, whisper, custom)")
    parser.add_argument("--url", help="HuggingFace repository ID or direct GGUF URL")
    parser.add_argument("--filename", help="Filename to download from HF repo")
    parser.add_argument("--gguf", help="Path to local .gguf file")
    parser.add_argument("--custom-script", help="Path to custom quantizer Python script")
    args = parser.parse_args()

    arch = args.arch
    if not arch:
        if args.alias in ("whisper", "whisper-base"):
            arch = "whisper"
        else:
            arch = args.alias

    source = "local" if args.gguf else "auto"
    run_setup(
        alias=args.alias,
        arch_or_quantizer=arch,
        source_type=source,
        url_or_repo=args.url,
        filename=args.filename,
        local_gguf_path=args.gguf,
        custom_script=args.custom_script
    )
