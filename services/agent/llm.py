"""Cliente LLM del orquestador. Los mensajes viajan como dicts con el formato de chat de OpenAI; el cliente real usa
Semantic Kernel (conector OpenAIChatCompletion) apuntando al endpoint /openai/v1/ del recurso de Azure AI Foundry.

No se fija la temperatura por defecto: algunos modelos de razonamiento rechazan valores distintos de 1. Si el modelo
desplegado la acepta, se fija con LLM_TEMPERATURE=0 para repetibilidad (scripts/check_llm.py lo comprueba)."""
import json
import os
import time
from dataclasses import dataclass, field


class LLMError(Exception):
    def __init__(self, message, status=None, retryable=False):
        super().__init__(message)
        self.message, self.status, self.retryable = message, status, retryable


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    raw_arguments: str = ""
    args_error: str | None = None          # JSON invalido en los argumentos que genero el modelo


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass
class LLMReply:
    text: str = ""
    tool_calls: list = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    finish_reason: str | None = None


def normalize_endpoint(endpoint):
    e = (endpoint or "").strip()
    if not e:
        raise LLMError("AZURE_OPENAI_ENDPOINT no configurado")
    return e if e.endswith("/") else e + "/"


def _usage_from(meta):
    u = (meta or {}).get("usage")
    if u is None:
        return Usage()
    g = lambda o, k: (o.get(k) if isinstance(o, dict) else getattr(o, k, None)) or 0
    pd, cd = g(u, "prompt_tokens_details"), g(u, "completion_tokens_details")
    return Usage(prompt_tokens=g(u, "prompt_tokens"), completion_tokens=g(u, "completion_tokens"),
                 cached_tokens=g(pd, "cached_tokens") if pd else 0, reasoning_tokens=g(cd, "reasoning_tokens") if cd else 0)


class SKChatClient:
    """Cliente real: Semantic Kernel + AsyncOpenAI contra la API v1 de Azure (el nombre del despliegue va como `model`)."""

    def __init__(self, endpoint, api_key, deployment, temperature=None, reasoning_effort=None, timeout_s=60.0, max_retries=2,
                 http_client=None):
        if not api_key or api_key.startswith(("REEMPLAZAR", "pega_aqui", "<")):
            raise LLMError("AZURE_OPENAI_API_KEY no configurada")
        if not deployment:
            raise LLMError("AZURE_OPENAI_DEPLOYMENT_NAME no configurado (tambien se acepta AZURE_OPENAI_DEPLOYMENT)")
        from openai import AsyncOpenAI
        from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
        self.deployment, self.temperature, self.reasoning_effort = deployment, temperature, reasoning_effort
        self._client = AsyncOpenAI(base_url=normalize_endpoint(endpoint), api_key=api_key, timeout=timeout_s,
                                   max_retries=max_retries, http_client=http_client)
        self._service = OpenAIChatCompletion(ai_model_id=deployment, async_client=self._client)

    @classmethod
    def from_env(cls, **kw):
        e = os.environ.get
        t = e("LLM_TEMPERATURE", "").strip()
        return cls(e("AZURE_OPENAI_ENDPOINT", ""), e("AZURE_OPENAI_API_KEY", ""), (e("AZURE_OPENAI_DEPLOYMENT_NAME") or e("AZURE_OPENAI_DEPLOYMENT") or ""),
                   temperature=float(t) if t else None, reasoning_effort=e("LLM_REASONING_EFFORT", "").strip() or None, **kw)

    # ---- conversion mensajes (dict estilo OpenAI) -> ChatHistory de SK
    @staticmethod
    def _history(messages):
        from semantic_kernel.contents import (AuthorRole, ChatHistory, ChatMessageContent, FunctionCallContent,
                                              FunctionResultContent, TextContent)
        h = ChatHistory()
        for m in messages:
            r = m["role"]
            if r == "system":
                h.add_message(ChatMessageContent(role=AuthorRole.SYSTEM, content=m["content"]))
            elif r == "user":
                h.add_message(ChatMessageContent(role=AuthorRole.USER, content=m["content"]))
            elif r == "assistant":
                items = [TextContent(text=m["content"])] if m.get("content") else []
                for c in m.get("tool_calls", []):
                    items.append(FunctionCallContent(id=c["id"], name=c["function"]["name"], arguments=c["function"]["arguments"]))
                h.add_message(ChatMessageContent(role=AuthorRole.ASSISTANT, items=items))
            elif r == "tool":
                h.add_message(ChatMessageContent(role=AuthorRole.TOOL, items=[FunctionResultContent(
                    id=m["tool_call_id"], name=m.get("name", "tool"), result=m["content"])]))
        return h

    async def complete(self, messages, tools=None):
        from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
        from semantic_kernel.contents import FunctionCallContent
        kw = {}
        if tools:
            kw.update(tools=tools, tool_choice="auto")
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        if self.reasoning_effort:
            kw["reasoning_effort"] = self.reasoning_effort
        settings = OpenAIChatPromptExecutionSettings(ai_model_id=self.deployment, **kw)
        t0 = time.time()
        try:
            msg = await self._service.get_chat_message_content(self._history(messages), settings)
        except Exception as e:                                          # el detalle del proveedor no se propaga tal cual
            cause = e.__cause__ or e
            status = getattr(cause, "status_code", None)
            detail = getattr(cause, "message", None) or str(cause)
            raise LLMError(f"{type(cause).__name__}: {detail[:300]}", status, retryable=status in (408, 429, 500, 502, 503, 504)) from e
        if msg is None:
            raise LLMError("el LLM no devolvio mensaje")
        calls = []
        for it in msg.items:
            if isinstance(it, FunctionCallContent):
                raw = it.arguments if isinstance(it.arguments, str) else json.dumps(it.arguments or {})
                try:
                    args = json.loads(raw) if raw else {}
                    err = None if isinstance(args, dict) else "los argumentos no son un objeto JSON"
                    args = args if isinstance(args, dict) else {}
                except ValueError:
                    args, err = {}, "los argumentos no son JSON valido"
                calls.append(ToolCall(it.id, it.name, args, raw, err))
        text = "".join(i.text for i in msg.items if i.__class__.__name__ == "TextContent" and i.text) or (msg.content or "")
        if calls and text == (msg.content or "") and not any(i.__class__.__name__ == "TextContent" for i in msg.items):
            text = ""
        return LLMReply(text=text.strip(), tool_calls=calls, usage=_usage_from(msg.metadata),
                        latency_s=time.time() - t0, finish_reason=getattr(msg, "finish_reason", None) and str(msg.finish_reason))


def assistant_message(reply):
    """Mensaje del asistente (dict) a partir de una respuesta, para reinyectarlo en la conversacion."""
    m = {"role": "assistant", "content": reply.text or None}
    if reply.tool_calls:
        m["tool_calls"] = [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.raw_arguments or json.dumps(c.arguments)}}
                           for c in reply.tool_calls]
    return m
