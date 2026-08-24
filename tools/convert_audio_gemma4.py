import sys
import numpy as np
from pathlib import Path

try:
    from gguf import GGUFReader
    from gguf.quants import dequantize
except ModuleNotFoundError:
    sys.path.append("/home/daino/llama-mtp/llama.cpp/gguf-py")
    from gguf import GGUFReader
    from gguf.quants import dequantize

def convert_audio_mmproj(mmproj_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n--- Reading audio tensors from {mmproj_path} to {out_dir} ---")
    reader = GGUFReader(mmproj_path)
    
    saved_count = 0
    for t in reader.tensors:
        # Filter for audio tensors (a.* or mm.a.*)
        if not (t.name.startswith("a.") or t.name.startswith("a_") or "audio" in t.name.lower() or t.name.startswith("mm.a.")):
            continue
            
        data = t.data
        if t.tensor_type != 0: # not F32
            data = dequantize(data, t.tensor_type)
        else:
            data = np.frombuffer(data, dtype=np.float32)
        
        # GGUF tensor shape is reversed in numpy
        shape = tuple(reversed(t.shape))
        arr = data.reshape(shape).astype(np.float32)
        
        safe_name = t.name.replace(".", "_") + ".npy"
        np.save(out_dir / safe_name, arr)
        print(f"Saved {safe_name:40s} shape={arr.shape} dtype={arr.dtype}")
        saved_count += 1
        
    print(f"Total audio tensors saved: {saved_count}")

if __name__ == "__main__":
    convert_audio_mmproj("/home/daino/llama-mtp/models/gemma-4-12b-it-mmproj-F16.gguf",
                         Path(__file__).resolve().parents[1] / "quantized_weights_gemma4" / "audio")
    convert_audio_mmproj("/home/daino/llama-mtp/models/gemma-4-E4B-it-mmproj-F16.gguf",
                         Path(__file__).resolve().parents[1] / "quantized_weights_gemma4-e4b" / "audio")
