"""
Base Quantizer interface and plugin loader for Alveare NPU model quantization.

All custom quantizers inherit from BaseQuantizer or implement a `main(gguf_path, out_dir)` function.
"""
import importlib.util
import json
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Root directory of the repository
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from tools.convert.gemv_q_convert import pack_to_combined, quantize_to_q4_0


class BaseQuantizer(ABC):
    """
    Abstract Base Class for Alveare Model Quantizers.
    Custom plugins can extend this class to implement custom quantization,
    tensor naming, padding, and metadata extraction logic.
    """

    def __init__(self, name: str = "custom"):
        self.name = name

    def pad_matrix(self, W: np.ndarray, target_N: int, target_K: int) -> np.ndarray:
        """Pad 2D matrix W to (target_N, target_K) with zeros."""
        N, K = W.shape
        if N == target_N and K == target_K:
            return W
        padded = np.zeros((target_N, target_K), dtype=W.dtype)
        padded[:N, :K] = W
        return padded

    def quantize_and_pack_tensor(
        self, W: np.ndarray, target_N: int, target_K: int
    ) -> np.ndarray:
        """Pad W to target shape, quantize to Q4_0 layout, and pack to uint8 format."""
        W_padded = self.pad_matrix(W, target_N, target_K)
        w_q4, scales = quantize_to_q4_0(W_padded)
        return pack_to_combined(w_q4, scales)

    @abstractmethod
    def quantize(self, gguf_path: str, out_dir: str) -> Dict[str, Any]:
        """
        Extract weights from GGUF, quantize, and write packed .npy files and config.json
        into out_dir.

        Returns:
            Dict containing the model configuration.
        """
        pass


def load_quantizer_plugin(quantizer_spec: str) -> Any:
    """
    Load a quantizer instance or script by architecture keyword or file path.

    Args:
        quantizer_spec: One of 'gemma4', 'gemma4-e4b', 'gemma3', 'llama',
                       or a path to a Python script (e.g. '/path/to/my_quantizer.py')

    Returns:
        An object with a `quantize(gguf_path, out_dir)` method or a module with main().
    """
    spec_path = Path(quantizer_spec)

    # 1. Custom Python file
    if spec_path.is_file() and spec_path.suffix == ".py":
        module_name = f"custom_quantizer_{spec_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, str(spec_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load custom quantizer from {quantizer_spec}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)

        # Look for a class inheriting from BaseQuantizer, or a main/quantize function
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseQuantizer)
                and attr is not BaseQuantizer
            ):
                return attr()

        if hasattr(mod, "quantize"):
            return mod
        if hasattr(mod, "main"):
            class ScriptAdapter:
                def quantize(self, gguf_path: str, out_dir: str):
                    return mod.main(gguf_path=gguf_path, out_dir=out_dir)
            return ScriptAdapter()

        raise AttributeError(
            f"Custom quantizer script {quantizer_spec} must define a BaseQuantizer subclass, "
            f"a `quantize(gguf_path, out_dir)` function, or a `main(gguf_path, out_dir)` function."
        )

    # 2. Built-in quantizers
    key = quantizer_spec.lower().strip()
    if key in ("gemma4", "gemma-4", "gemma4-12b"):
        from tools import quantize_gemma4
        return quantize_gemma4
    elif key in ("gemma4-e4b", "e4b", "gemma4e"):
        from tools import quantize_gemma4e
        return quantize_gemma4e
    elif key in ("gemma3", "gemma-3"):
        from tools import quantize_gemma
        return quantize_gemma
    elif key in ("llama", "llama3", "llama-3.2"):
        from tools import quantize_model
        return quantize_model
    else:
        raise ValueError(
            f"Unknown architecture or missing quantizer script: '{quantizer_spec}'. "
            f"Supported built-ins: gemma4, gemma4-e4b, gemma3, llama. Or pass a path to a .py script."
        )
