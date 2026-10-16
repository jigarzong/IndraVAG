"""LLM guionado para pruebas del orquestador y autoprueba del arnes de evaluacion. NO produce resultados de evaluacion: los
numeros reportados salen solo de ejecutar el set contra el LLM real."""
import json
import uuid

from .llm import LLMReply, ToolCall, Usage


class ScriptedLLM:
    """policy(messages, tools, n) -> dict(text=..., tools=[(nombre, args), ...]); n = numero de llamada (desde 0)."""

    def __init__(self, policy, prompt_tokens=100, completion_tokens=20):
        self.policy, self.pt, self.ct = policy, prompt_tokens, completion_tokens
        self.calls = []

    async def complete(self, messages, tools=None):
        n = len(self.calls)
        self.calls.append(dict(messages=list(messages), tools=[t["function"]["name"] for t in (tools or [])]))
        step = self.policy(messages, tools, n) or {}
        calls = []
        for name, args in step.get("tools", []):
            raw = args if isinstance(args, str) else json.dumps(args)
            calls.append(ToolCall("call_" + uuid.uuid4().hex[:8], name, {} if isinstance(args, str) else args, raw,
                                  "los argumentos no son JSON valido" if isinstance(args, str) else None))
        return LLMReply(text=step.get("text", ""), tool_calls=calls, usage=Usage(self.pt, self.ct, 0, 0), latency_s=0.01)


def sequence(*steps):
    """Politica que responde con los pasos en orden; el ultimo se repite si se agotan."""
    return lambda m, t, n: steps[min(n, len(steps) - 1)]
