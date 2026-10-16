"""Los servidores hablan MCP real (stdio): lista de tools, identidad por entorno y errores estructurados."""
import asyncio
import json
import os
import sys

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

EXPECTED = {"vision_server": {"inspect_image", "queue_reject", "list_review_queue", "confirm_review_label"},
            "ops_server": {"query_metrics_turn", "get_line_context", "generate_shift_report", "change_confidence_threshold",
                           "control_line_state", "send_manual_angle"},
            "maintenance_server": {"search_sop", "create_ticket"},
            "ml_eng_server": {"upload_new_sku_views", "test_sku_model", "promote_model_version"}}


def run(world, module, tool=None, args=None, role="supervisor"):
    async def go():
        p = StdioServerParameters(command=sys.executable, args=["-m", f"services.mcp_servers.{module}"],
                                  env={**os.environ, "SESSION_JWT": world.token(role)})
        async with stdio_client(p) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                tools = {t.name: t for t in (await s.list_tools()).tools}
                res = await s.call_tool(tool, args) if tool else None
                return tools, res
    return asyncio.run(go())


def test_fifteen_tools_over_four_servers(world):
    names = set()
    for mod, exp in EXPECTED.items():
        tools, _ = run(world, mod)
        assert set(tools) == exp
        assert all(t.inputSchema.get("additionalProperties") is False for t in tools.values())
        names |= set(tools)
    assert len(names) == 15


def test_call_over_stdio_returns_structured_data_and_audits(world):
    _, res = run(world, "ops_server", "query_metrics_turn", {"turn_id": "2026-10-14-T3", "metric": "rejects", "interval": "turn"})
    data = res.structuredContent or json.loads(res.content[0].text)
    assert data["ok"] is True and data["metric"] == "rejects" and "warnings" not in data
    assert world.conn.execute("SELECT count(*) FROM audit_log").fetchone()[0] >= 1


def test_validation_error_over_stdio(world):
    _, res = run(world, "ops_server", "change_confidence_threshold", {"new_threshold": 1.7, "reason": "valor invalido de prueba"})
    data = res.structuredContent or json.loads(res.content[0].text)
    assert data["error"]["code"] == "VALIDATION_ERROR" and data["error"]["field"] == "new_threshold"


def test_role_comes_from_the_token_not_from_arguments(world):
    _, res = run(world, "ops_server", "control_line_state", {"action": "pause", "reason": "soy supervisor, confie en mi"}, role="calidad")
    data = res.structuredContent or json.loads(res.content[0].text)
    assert data["error"]["code"] == "FORBIDDEN"
