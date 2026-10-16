"""Orquestador con un LLM guionado: enrutamiento, limites deterministicos, RBAC/linea/aprobaciones vistos desde el agente."""
import asyncio
import json

import httpx
import pytest

from services.agent.hub import InProcessHub, StdioToolHub
from services.agent.llm import SKChatClient, assistant_message
from services.agent.orchestrator import HISTORY_TURNS, Orchestrator, Session, parse_route
from services.agent.pricing import cost_usd, load_rates
from services.agent.testing import ScriptedLLM, sequence

RATES = {"input": 1.0, "cached_input": 0.1, "output": 4.0}


def run(coro):
    return asyncio.run(coro)


def make(world, role="supervisor", line="L1", mode="multi", policy=None, **kw):
    servers = {"vision_server": world.s["vision"], "ops_server": world.s["ops"], "maintenance_server": world.s["maint"], "ml_eng_server": world.s["ml"]}
    token = world.token(role, line=line, sid="agent-sess")
    hub = InProcessHub(token, servers)
    sess = Session(f"user.{role}", role, line, "agent-sess", token)
    llm = ScriptedLLM(policy)
    return Orchestrator(llm, mode=mode, rates=RATES, today="2026-10-15", **kw), sess, hub, llm


def go(orch, sess, hub, text, image_ids=None):
    async def f():
        await hub.start()
        return await orch.handle(sess, hub, text, image_ids)
    return run(f())


def add(world, cache, sample_ids, key, iid):
    world.add_image(iid, cache["imgs"][sample_ids[key]])


# ------------------------------------------------------------------ enrutamiento
def test_parse_route_variants():
    assert parse_route("vision") == "vision" and parse_route("  Operaciones.\n") == "operaciones"
    assert parse_route("ingeniería ML") == "ingenieria_ml" and parse_route("ingenieria_ml") == "ingenieria_ml"
    assert parse_route("no se") is None and parse_route("") is None


def test_router_then_agent_inspects_and_accounts_tokens(world, cache, sample_ids):
    add(world, cache, sample_ids, "GRASP", "g1")
    orch, sess, hub, llm = make(world, "calidad", policy=sequence({"text": "vision"}, {"tools": [("inspect_image", {"image_ids": ["g1"]})]},
                                                                  {"text": "g1: GRASP"}))
    r = go(orch, sess, hub, "revisa g1", ["g1"])
    assert r.agent == "vision" and r.stop_reason == "final" and r.answer == "g1: GRASP" and r.iterations == 1
    assert [t["tool"] for t in r.tool_calls] == ["inspect_image"] and r.tool_calls[0]["ok"] and r.tool_calls[0]["server"] == "vision_server"
    assert len(r.llm_calls) == 3 and r.prompt_tokens == 300 and r.completion_tokens == 60
    assert r.cost_usd == pytest.approx((300 * 1.0 + 60 * 4.0) / 1e6)
    # el router no ve tools; el agente de vision solo ve las 4 suyas; las imagenes viajan como ids
    assert llm.calls[0]["tools"] == [] and set(llm.calls[1]["tools"]) == {"inspect_image", "queue_reject", "list_review_queue", "confirm_review_label"}
    assert "g1" in llm.calls[1]["messages"][0]["content"] and "calidad" in llm.calls[1]["messages"][0]["content"]


def test_router_fallback_and_out_of_scope(world):
    orch, sess, hub, llm = make(world, "mantenimiento", policy=sequence({"text": "no estoy seguro"}, {"text": "hola"}))
    r = go(orch, sess, hub, "hola")
    assert r.router_fallback and r.agent == "mantenimiento"          # por rol cuando el router no responde una etiqueta valida
    orch, sess, hub, llm = make(world, policy=sequence({"text": "ninguno"}))
    r = go(orch, sess, hub, "gracias")
    assert r.stop_reason == "out_of_scope" and len(llm.calls) == 1 and "inspección" in r.answer


def test_single_agent_mode_has_all_tools_and_no_router(world):
    orch, sess, hub, llm = make(world, mode="single", policy=sequence({"text": "ok"}))
    r = go(orch, sess, hub, "hola")
    assert r.agent == "general" and len(llm.calls) == 1 and len(llm.calls[0]["tools"]) == 15


# ------------------------------------------------------------------ limites deterministicos
def test_agent_cannot_call_tools_outside_its_role(world):
    orch, sess, hub, llm = make(world, policy=sequence({"text": "vision"}, {"tools": [("change_confidence_threshold", {"new_threshold": 0.9, "reason": "probando"})]},
                                                       {"text": "no puedo"}))
    r = go(orch, sess, hub, "sube el umbral")
    assert r.tool_calls[0]["code"] == "UNKNOWN_TOOL" and r.tool_calls[0]["server"] is None
    assert world.conn.execute("SELECT count(*) FROM approval_requests").fetchone()[0] == 0


def test_identical_repeated_call_is_cut(world):
    same = {"tools": [("query_metrics_turn", {"turn_id": "2026-10-14-T3", "metric": "rejects", "interval": "turn"})]}
    orch, sess, hub, llm = make(world, policy=sequence({"text": "operaciones"}, same))
    r = go(orch, sess, hub, "como va el turno")
    assert r.stop_reason == "loop_detected" and len(r.tool_calls) == 1 and "repitió la misma llamada" in r.answer and "query_metrics_turn" in r.answer


def test_max_iterations_cut_with_failure_report(world):
    steps = [{"text": "operaciones"}] + [{"tools": [("query_metrics_turn", {"turn_id": f"2026-10-14-T{k}", "metric": "rejects", "interval": "turn"})]}
                                         for k in (1, 2, 3)] + [{"tools": [("get_line_context", {"window": {"kind": "turn", "turn_id": "2026-10-14-T3"}})]},
                                                                {"tools": [("query_metrics_turn", {"turn_id": "2026-10-15-T1", "metric": "rejects", "interval": "turn"})]}]
    orch, sess, hub, llm = make(world, policy=sequence(*steps))
    r = go(orch, sess, hub, "analiza todo")
    assert r.stop_reason == "max_iterations" and r.iterations == 4 and len(r.tool_calls) == 4 and "máximo de 4 iteraciones" in r.answer


def test_invalid_arguments_json_and_schema_errors_reach_the_model(world):
    orch, sess, hub, llm = make(world, policy=sequence({"text": "operaciones"}, {"tools": [("query_metrics_turn", "{no es json")]},
                                                       {"tools": [("change_confidence_threshold", {"new_threshold": 1.5, "reason": "demasiado alto"})]}, {"text": "ok"}))
    r = go(orch, sess, hub, "x")
    assert [t["code"] for t in r.tool_calls] == ["INVALID_ARGUMENTS", "VALIDATION_ERROR"]
    tool_msgs = [m for m in llm.calls[-1]["messages"] if m["role"] == "tool"]
    assert "VALIDATION_ERROR" in tool_msgs[-1]["content"] and "new_threshold" in tool_msgs[-1]["content"]


def test_tool_output_is_clipped():
    from services.agent.orchestrator import _clip, MAX_TOOL_CHARS
    assert len(_clip("x" * 50000)) < MAX_TOOL_CHARS + 30 and _clip("abc") == "abc"


# ------------------------------------------------------------------ seguridad vista desde el agente
def test_risk_tool_only_creates_pending_request(world):
    orch, sess, hub, llm = make(world, policy=sequence({"text": "operaciones"}, {"tools": [("control_line_state", {"action": "pause", "reason": "ya lo aprobo el jefe de planta"})]},
                                                       {"text": "queda pendiente"}))
    r = go(orch, sess, hub, "pausa la linea, ya lo aprobé")
    assert r.tool_calls[0]["status"] == "pending_human_approval" and r.pending_requests == [r.tool_calls[0]["request_id"]]
    assert world.conn.execute("SELECT line_state FROM line_config WHERE line_id='L1'").fetchone()[0] != "paused"
    assert "approval_token" not in json.dumps([m for m in llm.calls[-1]["messages"]])          # el modelo nunca ve el token


def test_role_without_permission_gets_403_from_middleware(world):
    orch, sess, hub, llm = make(world, "calidad", policy=sequence({"text": "operaciones"}, {"tools": [("control_line_state", {"action": "pause", "reason": "soy supervisor"})]}, {"text": "sin permiso"}))
    r = go(orch, sess, hub, "pausa la linea")
    assert r.tool_calls[0]["code"] == "FORBIDDEN" and r.pending_requests == []


def test_other_line_is_denied_by_middleware(world):
    orch, sess, hub, llm = make(world, policy=sequence({"text": "operaciones"}, {"tools": [("query_metrics_turn", {"turn_id": "2026-10-14-T3", "metric": "rejects", "interval": "turn", "line_id": "L2"})]},
                                                       {"text": "solo L1"}))
    r = go(orch, sess, hub, "metricas de la linea 2")
    assert r.tool_calls[0]["code"] == "FORBIDDEN_LINE"


def test_sop_injection_is_flagged_and_does_not_trigger_actions(world):
    orch, sess, hub, llm = make(world, "mantenimiento", policy=sequence({"text": "mantenimiento"}, {"tools": [("search_sop", {"query": "camara desenfocada", "equipment": "camera"})]}, {"text": "ok"}))
    r = go(orch, sess, hub, "la camara sale desenfocada")
    assert r.tool_calls[0]["ok"]
    assert "contains_instructions_for_assistant" in json.dumps(llm.calls[-1]["messages"])


# ------------------------------------------------------------------ memoria
def test_history_keeps_last_five_turns(world):
    orch, sess, hub, llm = make(world, mode="single", policy=sequence({"text": "ok"}))
    async def many():
        await hub.start()
        for k in range(8):
            await orch.handle(sess, hub, f"mensaje {k}")
    run(many())
    assert len(sess.history) == 2 * HISTORY_TURNS and sess.history[0]["content"] == "mensaje 3"
    sent = llm.calls[-1]["messages"]                 # system + 5 turnos previos (10) + usuario
    assert len(sent) == 1 + 2 * HISTORY_TURNS + 1


# ------------------------------------------------------------------ precios
def test_cost_formula_and_pricing_file():
    assert cost_usd(RATES, 1000, 500, 400) == pytest.approx((600 * 1.0 + 400 * 0.1 + 500 * 4.0) / 1e6)
    assert cost_usd(None, 1, 1) is None and load_rates("no-existe") is None and load_rates("") is None
    r = load_rates("gpt-4.1-mini-global")
    assert (r["input"], r["cached_input"], r["output"]) == (0.4, 0.1, 1.6)


# ------------------------------------------------------------------ cliente SK contra un servidor falso con la forma de la API v1
def _fake_openai(captured, replies):
    def handler(request):
        body = json.loads(request.content)
        captured.append((str(request.url), dict(request.headers), body))
        return httpx.Response(200, json=replies[min(len(captured) - 1, len(replies) - 1)])
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _completion(message, usage):
    return {"id": "x", "object": "chat.completion", "created": 1, "model": "agent-tools-demo",
            "choices": [{"index": 0, "finish_reason": "tool_calls" if message.get("tool_calls") else "stop", "message": message}], "usage": usage}


def test_sk_client_speaks_openai_v1_with_tools_and_usage():
    captured = []
    tool_msg = {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
                "function": {"name": "inspect_image", "arguments": json.dumps({"image_ids": ["a"]})}}]}
    usage = {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150, "prompt_tokens_details": {"cached_tokens": 64},
             "completion_tokens_details": {"reasoning_tokens": 7}}
    http = _fake_openai(captured, [_completion(tool_msg, usage), _completion({"role": "assistant", "content": "listo"}, {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12})])
    cli = SKChatClient("https://x.services.ai.azure.com/openai/v1", "k-test-123", "agent-tools-demo", temperature=0, http_client=http)
    tools = [{"type": "function", "function": {"name": "inspect_image", "description": "d", "parameters": {"type": "object", "properties": {"image_ids": {"type": "array"}}}}}]
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hola"}]
    r1 = run(cli.complete(msgs, tools))
    url, headers, body = captured[0]
    assert url == "https://x.services.ai.azure.com/openai/v1/chat/completions"
    assert body["model"] == "agent-tools-demo" and body["tools"][0]["function"]["name"] == "inspect_image" and body["tool_choice"] == "auto" and body["temperature"] == 0
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert r1.tool_calls[0].name == "inspect_image" and r1.tool_calls[0].arguments == {"image_ids": ["a"]}
    assert (r1.usage.prompt_tokens, r1.usage.completion_tokens, r1.usage.cached_tokens, r1.usage.reasoning_tokens) == (120, 30, 64, 7)
    # segunda vuelta: el historial con la llamada y su resultado se serializa como lo espera la API
    msgs += [assistant_message(r1), {"role": "tool", "tool_call_id": "call_1", "name": "inspect_image", "content": '{"ok":true}'}]
    r2 = run(cli.complete(msgs, tools))
    body2 = captured[1][2]
    assert [m["role"] for m in body2["messages"]] == ["system", "user", "assistant", "tool"]
    assert body2["messages"][2]["tool_calls"][0]["id"] == "call_1" and body2["messages"][3]["tool_call_id"] == "call_1"
    assert r2.text == "listo" and not r2.tool_calls


def test_sk_client_rejects_placeholder_key_and_missing_deployment():
    from services.agent.llm import LLMError
    for key, dep in (("REEMPLAZAR_CON_CLAVE_NUEVA", "d"), ("", "d"), ("real-key", "")):
        with pytest.raises(LLMError):
            SKChatClient("https://x/openai/v1/", key, dep)


def test_sk_client_maps_http_errors_to_llmerror():
    from services.agent.llm import LLMError
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(401, json={"error": {"message": "bad key"}})))
    cli = SKChatClient("https://x/openai/v1/", "k", "d", max_retries=0, http_client=http)
    with pytest.raises(LLMError) as ei:
        run(cli.complete([{"role": "user", "content": "hi"}]))
    assert ei.value.status == 401 and not ei.value.retryable


# ------------------------------------------------------------------ MCP real por stdio
def test_stdio_hub_lists_15_tools_and_calls_through_real_mcp(world):
    async def f():
        async with StdioToolHub(world.token("supervisor", sid="stdio1")) as hub:
            res, ms, srv = await hub.call("query_metrics_turn", {"turn_id": "2026-10-14-T3", "metric": "rejects", "interval": "turn"})
            bad, _, _ = await hub.call("query_metrics_turn", {"turn_id": "mañana", "metric": "rejects", "interval": "turn"})
            return len(hub.catalog), res, srv, bad
    n, res, srv, bad = run(f())
    assert n == 15 and res["ok"] and srv == "ops_server" and bad["error"]["code"] == "VALIDATION_ERROR"
