"""Orquestador: router -> agente especialista (ReAct) con limites deterministicos.

Limites que aplica el CODIGO (no el prompt):
  * maximo MAX_AGENT_ITERATIONS rondas Thought-Action-Observation por solicitud (por defecto 4); si el modelo sigue pidiendo tools,
    el flujo se corta y se entrega un informe de falla al usuario;
  * dos llamadas identicas (misma tool y mismos argumentos) en una solicitud = bucle: se corta sin ejecutar la repetida;
  * un agente solo puede llamar a las tools de su rol; el resto devuelve UNKNOWN_TOOL;
  * identidad, rol y linea viajan en el JWT del hub, nunca en los argumentos que genera el modelo.
"""
import asyncio
import json
import os
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field

from . import prompts
from .llm import LLMError, assistant_message
from .pricing import cost_usd, load_rates

HISTORY_TURNS = 5
MAX_TOOL_CHARS = 6000


@dataclass
class Session:
    user_id: str
    role: str
    line_id: str
    session_id: str
    token: str
    history: list = field(default_factory=list)      # ultimos HISTORY_TURNS turnos: [{"role","content"},...]
    last_agent: str | None = None

    @property
    def ident(self):
        return dict(user_id=self.user_id, role=self.role, line_id=self.line_id)


@dataclass
class RunResult:
    answer: str = ""
    agent: str | None = None
    stop_reason: str = "final"          # final | out_of_scope | max_iterations | loop_detected | llm_error | timeout
    iterations: int = 0
    tool_calls: list = field(default_factory=list)
    llm_calls: list = field(default_factory=list)
    router_fallback: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float | None = None
    latency_s: float = 0.0
    pending_requests: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _canon(o):
    return json.dumps(o, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def _strip(s):
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")


def parse_route(text):
    t = _strip(text or "")
    t = t.replace("ingenieria ml", "ingenieria_ml").replace("ingenieria-ml", "ingenieria_ml")
    m = re.search(r"\b(ingenieria_ml|ingenieria|vision|operaciones|mantenimiento|ninguno)\b", t)
    if not m:
        return None
    return "ingenieria_ml" if m.group(1) == "ingenieria" else m.group(1)


def _clip(s, n=MAX_TOOL_CHARS):
    return s if len(s) <= n else s[:n] + " ...[truncado]"


def _digest(res):
    if res.get("ok"):
        return "ok" + (f" status={res['status']}" if res.get("status") else "")
    e = res.get("error", {})
    return f"error {e.get('code')}"


class Orchestrator:
    def __init__(self, llm, mode="multi", max_iterations=None, rates=None, today=None, request_timeout_s=180.0):
        if mode not in ("multi", "single"):
            raise ValueError("mode debe ser 'multi' o 'single'")
        self.llm, self.mode = llm, mode
        self.max_iterations = int(max_iterations or os.environ.get("MAX_AGENT_ITERATIONS", 4))
        self.rates = rates if rates is not None else load_rates(os.environ.get("PRICING_MODEL_KEY"))
        self.today = today or os.environ.get("DEMO_TODAY", "2026-10-15")
        self.timeout_s = request_timeout_s

    # ------------------------------------------------------------------ API
    async def handle(self, session, hub, text, image_ids=None):
        run = RunResult()
        t0 = time.time()
        try:
            await asyncio.wait_for(self._run(run, session, hub, text, image_ids or []), self.timeout_s)
        except asyncio.TimeoutError:
            run.stop_reason = "timeout"
            run.answer = self._failure_report(run, "la solicitud superó el tiempo máximo permitido")
        except LLMError as e:
            run.stop_reason = "llm_error"
            run.answer = self._failure_report(run, f"el modelo de lenguaje no respondió ({e.message[:160]})")
        run.latency_s = round(time.time() - t0, 3)
        run.cost_usd = round(cost_usd(self.rates, run.prompt_tokens, run.completion_tokens, run.cached_tokens), 6) if self.rates else None
        self._remember(session, text, image_ids or [], run)
        return run

    # ------------------------------------------------------------------ nucleo
    async def _run(self, run, session, hub, text, image_ids):
        agent = await self._route(run, session, text, image_ids)
        run.agent = agent
        if agent == "ninguno":
            run.stop_reason, run.answer = "out_of_scope", prompts.OUT_OF_SCOPE
            return
        session.last_agent = agent
        if agent == "general":
            system = prompts.single_system(self.today, session.ident, image_ids)
            allowed = None
        else:
            system = prompts.agent_system(agent, self.today, session.ident, image_ids)
            allowed = set(prompts.TOOLS[agent])
        tools = hub.tool_schemas(allowed)
        msgs = [{"role": "system", "content": system}, *session.history, {"role": "user", "content": text}]
        seen = set()
        while True:
            reply = await self.llm.complete(msgs, tools)
            self._account(run, "agent", reply)
            if not reply.tool_calls:
                run.answer = reply.text or "No tengo una respuesta para esa solicitud."
                return
            if run.iterations >= self.max_iterations:
                run.stop_reason = "max_iterations"
                run.answer = self._failure_report(run, f"se alcanzó el máximo de {self.max_iterations} iteraciones sin terminar")
                return
            run.iterations += 1
            msgs.append(assistant_message(reply))
            for c in reply.tool_calls:
                key = (c.name, _canon(c.arguments))
                if key in seen:
                    run.stop_reason = "loop_detected"
                    run.answer = self._failure_report(run, f"el asistente repitió la misma llamada ({c.name}) sin avanzar")
                    return
                seen.add(key)
                if c.args_error:
                    res = {"ok": False, "error": {"code": "INVALID_ARGUMENTS", "message": c.args_error, "retryable": False, "http_status": 400}}
                    ms, server = 0.0, None
                else:
                    res, ms, server = await hub.call(c.name, c.arguments, allowed)
                run.tool_calls.append(dict(step=run.iterations, tool=c.name, args=c.arguments, server=server, ok=bool(res.get("ok")),
                                           code=None if res.get("ok") else res.get("error", {}).get("code"), latency_ms=ms,
                                           status=res.get("status"), request_id=res.get("request_id"),
                                           message=None if res.get("ok") else res.get("error", {}).get("message", "")[:160], result=res))
                if res.get("ok") and res.get("status") == "pending_human_approval" and res.get("request_id"):
                    run.pending_requests.append(res["request_id"])
                msgs.append({"role": "tool", "tool_call_id": c.id, "name": c.name, "content": _clip(json.dumps(res, ensure_ascii=False, default=str))})

    async def _route(self, run, session, text, image_ids):
        if self.mode == "single":
            return "general"
        extra = f"\n(El usuario adjuntó {len(image_ids)} imagen(es).)" if image_ids else ""
        msgs = [{"role": "system", "content": prompts.ROUTER},
                {"role": "user", "content": f"Especialista anterior: {session.last_agent or 'ninguno'}\nMensaje del usuario: {text}{extra}"}]
        reply = await self.llm.complete(msgs, None)
        self._account(run, "router", reply)
        label = parse_route(reply.text)
        if label is None:
            run.router_fallback = True
            label = prompts.ROLE_DEFAULT[session.role]
        return label

    # ------------------------------------------------------------------ utilidades
    @staticmethod
    def _account(run, who, reply):
        u = reply.usage
        run.prompt_tokens += u.prompt_tokens
        run.completion_tokens += u.completion_tokens
        run.cached_tokens += u.cached_tokens
        run.reasoning_tokens += u.reasoning_tokens
        run.llm_calls.append(dict(who=who, prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens, cached_tokens=u.cached_tokens,
                                  reasoning_tokens=u.reasoning_tokens, latency_s=round(reply.latency_s, 3), tool_calls=len(reply.tool_calls)))

    @staticmethod
    def _failure_report(run, why):
        lines = [f"No pude completar la solicitud: {why}.", "Esto fue lo que se ejecutó:"]
        if not run.tool_calls:
            lines.append("- (ninguna tool se ejecutó)")
        for t in run.tool_calls:
            a = _canon(t["args"])
            lines.append(f"- {t['tool']}({a[:100]}) -> {'ok' if t['ok'] else 'error ' + str(t['code'])}")
        lines.append("Escalo el caso a una persona: indícame cómo continuar o revísalo con el supervisor de turno.")
        return "\n".join(lines)

    def _remember(self, session, text, image_ids, run):
        user = text + (f" [imágenes adjuntas: {', '.join(image_ids)}]" if image_ids else "")
        ans = run.answer
        if run.tool_calls:
            dg = "; ".join(f"{t['tool']}({_canon(t['args'])[:70]})={_digest({'ok': t['ok'], 'status': t['status'], 'error': {'code': t['code']}})}"
                           for t in run.tool_calls[:6])
            ans += f"\n[acciones registradas por el sistema: {dg}]"
        session.history += [{"role": "user", "content": user}, {"role": "assistant", "content": ans}]
        del session.history[:-2 * HISTORY_TURNS]
