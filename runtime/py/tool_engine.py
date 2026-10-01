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

WEB_SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Cerca informazioni aggiornate sul web (notizie, meteo, eventi, curiosità, definizioni) tramite DuckDuckGo e Wikipedia senza necessità di chiavi API.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "La query di ricerca da inviare al motore di ricerca (es. 'meteo Firenze oggi', 'chi ha vinto la Formula 1', 'ultime notizie')."
                }
            },
            "required": ["query"]
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
        """Registers built-in tools like safe terminal bash and web search."""
        self.register_tool(
            name="bash",
            schema=BASH_TOOL_SCHEMA,
            handler=self._handle_bash
        )
        self.register_tool(
            name="web_search",
            schema=WEB_SEARCH_TOOL_SCHEMA,
            handler=self._handle_web_search
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

    def get_tools_system_prompt(self) -> str:
        """Returns concise instructions explaining available tools to the LLM."""
        lines = [
            "\nStrumenti disponibili che puoi invocare:",
            "- web_search(query=\"...\"): Cerca su Internet per meteo, notizie recenti, informazioni aggiornate o fatti.",
            "- bash(command=\"...\"): Esegue un comando da terminale Linux in sandbox sicura.",
            "\nPer invocare uno strumento quando necessario (es. meteo, ricerche sul web, comandi), scrivi ESATTAMENTE:",
            "[TOOL_CALL: nome_tool({\"arg\": \"valore\"})]",
            "Esempio: [TOOL_CALL: web_search({\"query\": \"meteo Firenze\"})]",
            "Quando invochi uno strumento non aggiungere altro testo; riceverai il risultato per formulare la risposta finale."
        ]
        return "\n".join(lines)

    def _handle_bash(self, args: Dict[str, Any]) -> Dict[str, Any]:
        cmd = args.get("command", "")
        if not cmd:
            return {"error": "Nessun comando specificato."}
        sandbox = get_sandbox()
        result = sandbox.execute(cmd)
        return result

    def _handle_web_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        query = args.get("query", "")
        if not query:
            return {"error": "Nessuna query di ricerca specificata."}
        from runtime.py.web_search import get_web_search_engine
        engine = get_web_search_engine()
        return engine.search(query, max_results=3)

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
