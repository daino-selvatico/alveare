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

def convert_mmproj(mmproj_path="/home/daino/llama-mtp/models/gemma-4-12b-it-mmproj-F16.gguf",
                   out_dir=None):
    if out_dir is None:
        out_dir = Path(__file__).resolve().parents[1] / "quantized_weights_gemma4" / "vision"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading mmproj from {mmproj_path}...")
    reader = GGUFReader(mmproj_path)
    
    for t in reader.tensors:
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
        
    print(f"\nAll vision embedder tensors saved to {out_dir}")

if __name__ == "__main__":
    convert_mmproj()
