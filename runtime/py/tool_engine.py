"""
Tool Calling and Execution Engine for Alveare.
Integrates safe terminal execution via SandboxEngine and provides extensible MCP (Model Context Protocol) tool support.
"""
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from runtime.py.sandbox import get_sandbox

logger = logging.getLogger("alveare.tools")

BASH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Esegue un comando da terminale Linux in una sandbox sicura e protetta (Bubblewrap). Il filesystem di sistema è in sola lettura, mentre la cartella workspace è accessibile in lettura/scrittura. Utilizzalo per calcoli, interrogare comandi di sistema, script o verificare informazioni.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Il comando di shell da eseguire (es. 'uptime', 'date', 'curl ...', 'cat file')."
                }
            },
            "required": ["command"]
        }
    }
}


class ToolEngine:
    """Manages tool registration, schema exportation, and execution."""

    def __init__(self):
        self._tools: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
        self._schemas: List[Dict[str, Any]] = []
        self._register_default_tools()

    def _register_default_tools(self):
        """Registers built-in tools like safe terminal bash."""
        self.register_tool(
            name="bash",
            schema=BASH_TOOL_SCHEMA,
            handler=self._handle_bash
        )

    def register_tool(
        self,
        name: str,
        schema: Dict[str, Any],
        handler: Callable[[Dict[str, Any]], Dict[str, Any]]
    ):
        self._tools[name] = handler
        # Remove old schema if exists
        self._schemas = [s for s in self._schemas if s.get("function", {}).get("name") != name]
        self._schemas.append(schema)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Returns OpenAPI/OpenAI-compatible tool definitions."""
        return list(self._schemas)

    def _handle_bash(self, args: Dict[str, Any]) -> Dict[str, Any]:
        cmd = args.get("command", "")
        if not cmd:
            return {"error": "Nessun comando specificato."}
        sandbox = get_sandbox()
        result = sandbox.execute(cmd)
        return result

    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a registered tool by name with provided arguments."""
        if name not in self._tools:
            return {
                "error": f"Tool '{name}' non riconosciuto.",
                "available_tools": list(self._tools.keys())
            }

        try:
            handler = self._tools[name]
            return handler(arguments)
        except Exception as e:
            logger.error(f"Errore durante l'esecuzione del tool '{name}': {e}", exc_info=True)
            return {"error": f"Errore interno tool '{name}': {str(e)}"}


# Global singleton instance
_GLOBAL_TOOL_ENGINE: Optional[ToolEngine] = None

def get_tool_engine() -> ToolEngine:
    global _GLOBAL_TOOL_ENGINE
    if _GLOBAL_TOOL_ENGINE is None:
        _GLOBAL_TOOL_ENGINE = ToolEngine()
    return _GLOBAL_TOOL_ENGINE
