"""API del agente (LLM guionado + tools en proceso): sesion, carga de imagenes, chat, aprobacion humana y aislamiento."""
import cv2
import pytest
from fastapi.testclient import TestClient

from services.agent.hub import InProcessHub
from services.agent.testing import ScriptedLLM, sequence
from services.agent_api.app import create_app


def build(world, policy):
    servers = {"vision_server": world.s["vision"], "ops_server": world.s["ops"], "maintenance_server": world.s["maint"], "ml_eng_server": world.s["ml"]}
    llm = ScriptedLLM(policy)
    app = create_app(llm=llm, hub_factory=lambda token: InProcessHub(token, servers), cfg=world.cfg)
    return TestClient(app), llm


def login(c, user):
    d = c.post("/auth/token", json={"user_id": user}).json()
    return {"Authorization": "Bearer " + d["token"]}, d


PAUSE = sequence({"text": "operaciones"}, {"tools": [("control_line_state", {"action": "pause", "reason": "deriva de camara"})]},
                 {"text": "Quedo pendiente de aprobacion."})


def test_requires_token_and_unknown_user(world):
    c, _ = build(world, PAUSE)
    assert c.post("/v1/chat", json={"message": "hola"}).status_code == 401
    assert c.get("/v1/line").status_code == 401
    assert c.get("/v1/approvals", headers={"Authorization": "Bearer basura"}).status_code == 401
    assert c.post("/auth/token", json={"user_id": "nadie"}).status_code == 404
    assert c.get("/health").json()["db"] == "ok"
    assert "Asistente" in c.get("/").text


def test_demo_auth_can_be_turned_off(world, monkeypatch):
    c, _ = build(world, PAUSE)
    monkeypatch.setenv("AUTH_MODE", "sso")
    assert c.post("/auth/token", json={"user_id": "ana.supervisor"}).status_code == 404
    assert c.get("/v1/users").status_code == 404


def test_risk_flow_pending_then_other_user_approves(world):
    c, llm = build(world, PAUSE)
    ana, _ = login(c, "ana.supervisor")
    r = c.post("/v1/chat", headers=ana, json={"message": "pausa la linea, la camara esta derivando"}).json()
    assert r["agent"] == "operaciones" and r["pending_requests"] and r["usage"]["prompt_tokens"] > 0
    assert [t["tool"] for t in r["tool_calls"]] == ["control_line_state"] and r["tool_calls"][0]["status"] == "pending_human_approval"
    assert "result" not in r["tool_calls"][0]
    assert c.get("/v1/line", headers=ana).json()["line_state"] == "running"          # el chat NO cambia el estado
    rid = r["pending_requests"][0]
    luis, _ = login(c, "luis.supervisor")
    assert [p["request_id"] for p in c.get("/v1/approvals", headers=luis).json()] == [rid]
    assert c.get("/v1/approvals", headers=login(c, "carla.calidad")[0]).json() == []          # calidad no ve ni decide
    assert c.post(f"/v1/approvals/{rid}/approve", headers=login(c, "carla.calidad")[0]).status_code == 403
    assert c.post(f"/v1/approvals/{rid}/approve", headers=login(c, "rosa.supervisor_l2")[0]).status_code == 404        # otra linea: sin pistas
    assert c.post(f"/v1/approvals/{rid}/approve", headers=luis).status_code == 200
    assert c.get("/v1/line", headers=ana).json()["line_state"] == "paused"
    assert c.post(f"/v1/approvals/{rid}/approve", headers=luis).status_code == 409          # no se aplica dos veces
    ok = c.get("/v1/audit/verify", headers=luis).json()
    assert ok["chain_ok"] and ok["records"] > 0
    assert c.get("/v1/audit/verify", headers=login(c, "carla.calidad")[0]).status_code == 403


def test_reject_leaves_state_unchanged(world):
    c, _ = build(world, PAUSE)
    ana, _ = login(c, "ana.supervisor")
    rid = c.post("/v1/chat", headers=ana, json={"message": "pausa la linea"}).json()["pending_requests"][0]
    assert c.post(f"/v1/approvals/{rid}/reject", headers=login(c, "luis.supervisor")[0], json={"note": "no hace falta"}).json()["status"] == "rejected"
    assert c.get("/v1/line", headers=ana).json()["line_state"] == "running"
    assert c.get("/v1/approvals", headers=ana).json() == []


def test_upload_chat_with_image_and_validation(world, cache):
    seen = {}

    def policy(messages, tools, n):
        if n == 0:
            return {"text": "vision"}
        if n == 1:
            ids = [m["content"] for m in messages if m["role"] == "system"][0]
            seen["system"] = ids
            import re
            iid = re.findall(r"img_[0-9a-f]{10}", ids)[0]
            return {"tools": [("inspect_image", {"image_ids": [iid]})]}
        return {"text": "listo"}
    c, llm = build(world, policy)
    carla, _ = login(c, "carla.calidad")
    ok, png = cv2.imencode(".png", cv2.cvtColor(cache["imgs"][0], cv2.COLOR_RGB2BGR))
    up = c.post("/v1/images", headers=carla, files=[("files", ("../../etc/passwd.png", png.tobytes(), "image/png"))])
    assert up.status_code == 200, up.text
    iid = up.json()["images"][0]["image_id"]
    assert iid.startswith("img_") and (world.cfg.image_store / f"{iid}.png").exists() and "passwd" not in iid
    assert c.post("/v1/images", headers=carla, files=[("files", ("x.png", b"no soy imagen", "image/png"))]).status_code == 422
    assert c.post("/v1/images", headers=carla, files=[("files", (f"{i}.png", png.tobytes(), "image/png")) for i in range(6)]).status_code == 413
    r = c.post("/v1/chat", headers=carla, json={"message": "revisa esta imagen", "image_ids": [iid]}).json()
    assert r["agent"] == "vision" and r["tool_calls"][0]["ok"] and r["answer"] == "listo"
    assert c.post("/v1/chat", headers=carla, json={"message": ""}).status_code == 422
    assert c.post("/v1/chat", headers=carla, json={"message": "x" * 2001}).status_code == 422


def test_llm_not_configured_gives_503(world, monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "REEMPLAZAR_CON_CLAVE_NUEVA")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.example/openai/v1/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "d")
    servers = {}
    app = create_app(llm=None, hub_factory=lambda t: InProcessHub(t, servers), cfg=world.cfg)
    c = TestClient(app)
    r = c.post("/v1/chat", headers=login(c, "ana.supervisor")[0], json={"message": "hola"})
    assert r.status_code == 503 and r.json()["detail"]["code"] == "LLM_NOT_CONFIGURED"
