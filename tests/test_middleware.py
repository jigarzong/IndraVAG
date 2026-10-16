"""Middleware comun: identidad, RBAC, aislamiento de linea, esquema JSON, idempotencia y auditoria."""
import json

import jwt
import pytest

from services.common import audit, auth


def test_missing_or_forged_token(world):
    assert world.s["ops"].invoke("query_metrics_turn", {}, None)["error"]["code"] == "UNAUTHENTICATED"
    forged = jwt.encode({"sub": "x", "role": "supervisor", "line_id": "L1", "sid": "s", "exp": 9999999999}, "otro-secreto", algorithm="HS256")
    assert world.s["ops"].invoke("query_metrics_turn", {}, forged)["error"]["code"] == "UNAUTHENTICATED"


def test_expired_token(world):
    t = auth.issue_token(world.cfg.jwt_secret, "u", "supervisor", "L1", "s", ttl_s=-10)
    assert world.s["ops"].invoke("query_metrics_turn", {}, t)["error"]["code"] == "UNAUTHENTICATED"


def test_rbac_blocks_wrong_role(world):
    r = world.call("ops", "control_line_state", {"action": "pause", "reason": "prueba de permisos"}, role="calidad")
    assert r["error"]["code"] == "FORBIDDEN" and r["error"]["http_status"] == 403
    assert world.conn.execute("SELECT count(*) FROM approval_requests").fetchone()[0] == 0          # ni siquiera se crea la solicitud


def test_unknown_tool(world):
    r = world.call("ops", "delete_everything", {})
    assert r["error"]["code"] == "UNKNOWN_TOOL"


def test_other_line_is_denied_and_isolated(world):
    r = world.call("ops", "get_line_context", {"line_id": "L2", "window": {"kind": "turn", "turn_id": "2026-10-14-T1"}})
    assert r["error"]["code"] == "FORBIDDEN_LINE"
    r = world.call("ops", "query_metrics_turn", {"turn_id": "2026-10-14-T1", "metric": "throughput", "interval": "turn"}, line="L2")
    assert r["ok"] and r["value"] != world.call("ops", "query_metrics_turn", {"turn_id": "2026-10-14-T1", "metric": "throughput", "interval": "turn"})["value"]


@pytest.mark.parametrize("args,field", [({"new_threshold": 1.5, "reason": "valor imposible"}, "new_threshold"),
                                        ({"new_threshold": 0.5}, ""),
                                        ({"new_threshold": 0.5, "reason": "ok ok ok", "extra": 1}, "")])
def test_schema_validation(world, args, field):
    r = world.call("ops", "change_confidence_threshold", args)
    assert r["error"]["code"] == "VALIDATION_ERROR" and r["error"]["http_status"] == 422
    if field:
        assert r["error"]["field"] == field
    assert world.conn.execute("SELECT count(*) FROM approval_requests").fetchone()[0] == 0          # la tool no se ejecuta


def test_idempotent_write_replays_same_result(world):
    a = {"issue_desc": "Banda de salida con vibracion fuerte al arrancar", "priority": "media", "equipment": "conveyor"}
    r1 = world.call("maint", "create_ticket", a, role="mantenimiento")
    r2 = world.call("maint", "create_ticket", a, role="mantenimiento")
    assert r1["ok"] and r2["ticket_id"] == r1["ticket_id"] and r2.get("idempotent_replay") is True
    assert world.conn.execute("SELECT count(*) FROM tickets WHERE ticket_id=?", (r1["ticket_id"],)).fetchone()[0] == 1


def test_errors_are_not_cached_as_idempotent(world):
    a = {"image_id": "img_x", "reason": "no inspeccionada todavia"}
    assert world.call("vision", "queue_reject", a)["error"]["code"] == "NOT_INSPECTED"
    assert world.call("vision", "queue_reject", a)["error"]["code"] == "NOT_INSPECTED"
    assert world.conn.execute("SELECT count(*) FROM idempotency").fetchone()[0] == 0


def test_audit_chain_and_content(world):
    world.call("ops", "query_metrics_turn", {"turn_id": "2026-10-14-T1", "metric": "rejects", "interval": "turn"})
    world.call("ops", "control_line_state", {"action": "pause", "reason": "x"}, role="calidad")           # denegada: tambien se audita
    ok, n, bad = audit.verify_chain(world.conn)
    assert ok and n >= 2
    rows = [json.loads(r["record_json"]) for r in world.conn.execute("SELECT record_json FROM audit_log")]
    assert {"user", "role", "line", "session", "tool", "latency_ms", "ok", "tool_version"} <= set(rows[0])
    assert any(r.get("code") == "FORBIDDEN" for r in rows)
    assert "token" not in world.conn.execute("SELECT group_concat(record_json) FROM audit_log").fetchone()[0].lower().replace("approval", "")


def test_audit_is_append_only_and_tamper_evident(world):
    world.call("ops", "query_metrics_turn", {"turn_id": "2026-10-14-T1", "metric": "rejects", "interval": "turn"})
    with pytest.raises(Exception, match="solo adicion"):
        world.conn.execute("UPDATE audit_log SET record_json='{}'")
    with pytest.raises(Exception, match="solo adicion"):
        world.conn.execute("DELETE FROM audit_log")
    world.conn.execute("DROP TRIGGER audit_no_update")                         # un atacante con acceso a la base debe dejar rastro
    world.conn.execute("UPDATE audit_log SET record_json='{\"tool\":\"otra\"}' WHERE seq=1")
    ok, n, bad = audit.verify_chain(world.conn)
    assert ok is False and bad == 1
