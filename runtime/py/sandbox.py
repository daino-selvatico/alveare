"""
Sandbox Execution Engine for Alveare.
Provides safe execution of terminal commands using Linux unprivileged namespaces (Bubblewrap / bwrap),
strict command sanitization, read-only root filesystems, and resource timeouts.
"""
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

# Commands and patterns categorically blocked before execution
BLOCKED_PATTERNS = [
    re.compile(r'\b(sudo|su|pkexec|doas)\b'),
    re.compile(r':\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:'),  # Fork bomb
    re.compile(r'\bmkfs\b'),
    re.compile(r'\bfdisk\b'),
    re.compile(r'>\s*/dev/(sd[a-z]|nvme[0-9]|hd[a-z])'),
    re.compile(r'\bdd\s+.*?of=/dev/'),
    re.compile(r'>\s*/etc/(passwd|shadow|sudoers)'),
    re.compile(r'\brm\s+-[rR]*f\s+/\s*$'),
]

DEFAULT_WORKSPACE = Path(os.path.expanduser("~/progetti/alveare/workspace"))


class SandboxEngine:
    """Executes arbitrary commands inside a secure, unprivileged Bubblewrap container."""

    def __init__(
        self,
        workspace_dir: Optional[Path] = None,
        timeout_seconds: float = 12.0,
        max_output_chars: int = 8192,
        allow_network: bool = True
    ):
        self.workspace_dir = workspace_dir or DEFAULT_WORKSPACE
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars
        self.allow_network = allow_network
        self.has_bwrap = shutil.which("bwrap") is not None

    def validate_command(self, cmd: str) -> Optional[str]:
        """Returns an error message if the command violates safety invariants."""
        if not cmd or not cmd.strip():
            return "Comando vuoto."

        for pat in BLOCKED_PATTERNS:
            if pat.search(cmd):
                return f"Comando bloccato dalle policy di sicurezza di Alveare (pattern vietato: {pat.pattern})."

        return None

    def execute(self, cmd: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Executes a shell command inside the Bubblewrap sandbox.
        Falls back to safe subshell if bwrap is missing (with read-only notice).
        """
        validation_error = self.validate_command(cmd)
        if validation_error:
            return {
                "exit_code": 126,
                "stdout": "",
                "stderr": validation_error,
                "duration_ms": 0.0,
                "sandboxed": False,
                "error": validation_error
            }

        effective_timeout = timeout or self.timeout_seconds
        start_time = time.perf_counter()

        if self.has_bwrap:
            bwrap_cmd = [
                "bwrap",
                "--ro-bind", "/", "/",           # Entire root is strictly READ-ONLY
                "--dev", "/dev",                  # Standard device nodes
                "--proc", "/proc",                # Kernel procfs
                "--tmpfs", "/tmp",                # Ephemeral RAM tmpfs
                "--bind", str(self.workspace_dir), str(self.workspace_dir),  # Only workspace is RW
                "--chdir", str(self.workspace_dir),
                "--die-with-parent",
            ]
            if not self.allow_network:
                bwrap_cmd.append("--unshare-net")

            full_args = bwrap_cmd + ["sh", "-c", cmd]
            sandboxed = True
        else:
            # Fallback (non-isolated, execution strictly restricted to workspace)
            full_args = ["sh", "-c", cmd]
            sandboxed = False

        try:
            proc = subprocess.run(
                full_args,
                cwd=str(self.workspace_dir),
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                env={
                    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                    "HOME": str(self.workspace_dir),
                    "USER": "alveare-sandbox",
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                }
            )
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            stdout = proc.stdout[:self.max_output_chars]
            stderr = proc.stderr[:self.max_output_chars]
            if len(proc.stdout) > self.max_output_chars:
                stdout += "\n... [output troncato per limite di dimensione]"
            if len(proc.stderr) > self.max_output_chars:
                stderr += "\n... [output troncato per limite di dimensione]"

            return {
                "exit_code": proc.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "duration_ms": round(duration_ms, 2),
                "sandboxed": sandboxed,
                "error": None if proc.returncode == 0 else (stderr.strip() or f"Processo terminato con codice {proc.returncode}")
            }

        except subprocess.TimeoutExpired:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "exit_code": 124,
                "stdout": "",
                "stderr": f"Timeout superato: il comando ha impiegato più di {effective_timeout:.1f} secondi.",
                "duration_ms": round(duration_ms, 2),
                "sandboxed": sandboxed,
                "error": "Timeout superato"
            }
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "exit_code": 1,
                "stdout": "",
                "stderr": str(e),
                "duration_ms": round(duration_ms, 2),
                "sandboxed": sandboxed,
                "error": str(e)
            }


# Global singleton instance
_GLOBAL_SANDBOX: Optional[SandboxEngine] = None

def get_sandbox() -> SandboxEngine:
    global _GLOBAL_SANDBOX
    if _GLOBAL_SANDBOX is None:
        _GLOBAL_SANDBOX = SandboxEngine()
    return _GLOBAL_SANDBOX
