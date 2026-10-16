"""API del agente. Endpoints (JWT Bearer salvo /health, /auth/token y la pantalla):

  POST /auth/token            (solo AUTH_MODE=demo) emite el JWT de un usuario demo
  POST /v1/images             carga de 1 a 5 imagenes -> image_id (el agente solo ve ids)
  POST /v1/chat               mensaje al asistente (router + agentes); mantiene la memoria de 5 turnos por sesion
  GET  /v1/line               estado de la linea (umbral, estado, version de modelo)
  GET  /v1/approvals          solicitudes pendientes que este usuario puede decidir
  POST /v1/approvals/{id}/approve | /reject    decision humana: aqui, y solo aqui, se emite el token de aprobacion
  GET  /v1/audit/verify       verifica la cadena de hashes de la auditoria
  GET  /health                estado del API, de la base y del servicio de vision

Arranque:  uvicorn --factory services.agent_api.app:create_app --port 8000

La aprobacion NO es una tool MCP: el LLM no tiene forma de llamarla ni de ver el token.
"""
import asyncio
import hashlib
import os
import threading
import time
import uuid
from pathlib import Path

import cv2
import httpx
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from services.agent.hub import HttpToolHub, InProcessHub, StdioToolHub
from services.agent.orchestrator import Orchestrator, Session
from services.common import audit, auth, db, effects
from services.common.approvals import ApprovalError, ApprovalService
from services.common.config import Config
from services.vision_api.quality import assess
from services.vision_api.settings import Settings as VisionSettings

from .users import DEMO_USERS

STATIC = Path(__file__).parent / "static"
SESSION_TTL_S = 4 * 3600
MAX_UPLOAD_BYTES = int(os.environ.get("VISION_MAX_IMAGE_BYTES", 5 * 1024 * 1024))
MAX_IMAGES = int(os.environ.get("VISION_MAX_IMAGES_PER_REQUEST", 5))


class TokenRequest(BaseModel):
    user_id: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    image_ids: list[str] = Field(default_factory=list, max_length=5)


class RejectRequest(BaseModel):
    note: str = Field(default="", max_length=200)


def default_hub_factory():
    kind = os.environ.get("AGENT_HUB", "http")
    if kind == "http":
        return lambda token: HttpToolHub.from_env(token)
    if kind == "stdio":
        return lambda token: StdioToolHub(token)
    if kind == "inprocess":
        return lambda token: InProcessHub(token)
    raise RuntimeError(f"AGENT_HUB desconocido: {kind}")


def create_app(llm=None, hub_factory=None, cfg=None):
    cfg = cfg or Config.load()
    if len(cfg.jwt_secret) < 32 or len(cfg.approval_secret) < 32:
        raise RuntimeError("JWT_SECRET y APPROVAL_SECRET deben tener al menos 32 caracteres (python scripts/init_env.py los genera)")
    app = FastAPI(title="Asistente agentico de operacion de vision", version="1.0")
    conn = db.init_schema(db.connect(cfg.db_path))
    approvals = ApprovalService(conn, cfg)
    approvals.executors = effects.EXECUTORS
    db_lock = threading.Lock()
    sessions, session_locks = {}, {}
    state = dict(llm=llm, orch=None)
    hub_factory = hub_factory or default_hub_factory()

    def get_orch():
        if state["orch"] is None:
            if state["llm"] is None:
                from services.agent.llm import SKChatClient
                state["llm"] = SKChatClient.from_env()
            state["orch"] = Orchestrator(state["llm"], mode=os.environ.get("AGENT_MODE", "multi"))
        return state["orch"]

    def identity(request: Request):
        h = request.headers.get("authorization", "")
        try:
            return auth.verify_token(cfg.jwt_secret, h[7:] if h.lower().startswith("bearer ") else None)
        except auth.AuthError as e:
            raise HTTPException(401, dict(code="UNAUTHENTICATED", message=e.message))

    @app.exception_handler(ApprovalError)
    async def _approval_error(request, e):
        return JSONResponse(status_code=e.status, content=dict(detail=dict(code=e.code, message=e.message)))

    # ------------------------------------------------------------------ sesion
    @app.post("/auth/token")
    def token(req: TokenRequest):
        if os.environ.get("AUTH_MODE", "demo") != "demo":
            raise HTTPException(404, "no disponible")
        u = DEMO_USERS.get(req.user_id)
        if not u:
            raise HTTPException(404, dict(code="UNKNOWN_USER", message="usuario demo desconocido"))
        sid = "S-" + uuid.uuid4().hex[:12]
        return dict(token=auth.issue_token(cfg.jwt_secret, req.user_id, u["role"], u["line_id"], sid, ttl_s=8 * 3600), user_id=req.user_id,
                    role=u["role"], line_id=u["line_id"], session_id=sid, name=u["name"])

    @app.get("/v1/users")
    def users():
        if os.environ.get("AUTH_MODE", "demo") != "demo":
            raise HTTPException(404, "no disponible")
        return [dict(user_id=k, **v) for k, v in DEMO_USERS.items()]

    # ------------------------------------------------------------------ imagenes
    @app.post("/v1/images")
    async def upload(files: list[UploadFile] = File(...), ident: auth.Identity = Depends(identity)):
        if not files or len(files) > MAX_IMAGES:
            raise HTTPException(413, dict(code="TOO_MANY_IMAGES", message=f"de 1 a {MAX_IMAGES} imagenes por carga"))
        out = []
        for f in files:
            raw = await f.read()
            if len(raw) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, dict(code="IMAGE_TOO_LARGE", message=f"{f.filename}: supera {MAX_UPLOAD_BYTES} bytes"))
            bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) if raw else None
            if bgr is None:
                raise HTTPException(422, dict(code="INVALID_IMAGE", message=f"{f.filename}: no se pudo decodificar como imagen"))
            iid = "img_" + uuid.uuid4().hex[:10]
            ok, png = cv2.imencode(".png", bgr)                      # se guarda re-codificada: nunca el archivo original ni su nombre
            cfg.image_store.mkdir(parents=True, exist_ok=True)
            (cfg.image_store / f"{iid}.png").write_bytes(png.tobytes())
            rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (128, 128), interpolation=cv2.INTER_AREA)
            q = assess(rgb, min(bgr.shape[:2]), VisionSettings.load())
            with db_lock:
                conn.execute("INSERT INTO images VALUES(?,?,?,?,?,?,?,?,?)", (iid, ident.line_id, f"{iid}.png", hashlib.sha256(png.tobytes()).hexdigest(),
                                                                              ident.user_id, audit.now_iso(), None, int(q["ok"]), ",".join(q["flags"])))
                audit.append(conn, dict(event="image_uploaded", image_id=iid, user=ident.user_id, line=ident.line_id, quality_ok=q["ok"], bytes=len(png)))
            out.append(dict(image_id=iid, quality_ok=q["ok"], quality_flags=q["flags"]))
        return dict(images=out)

    # ------------------------------------------------------------------ chat
    @app.post("/v1/chat")
    async def chat(request: Request, req: ChatRequest, ident: auth.Identity = Depends(identity)):
        h = request.headers["authorization"][7:]
        now = time.time()
        for sid in [s for s, (t, _) in sessions.items() if now - t > SESSION_TTL_S]:
            sessions.pop(sid, None); session_locks.pop(sid, None)
        key = ident.session_id
        sess = sessions.get(key, (0, None))[1] or Session(ident.user_id, ident.role, ident.line_id, ident.session_id, h)
        sess.token = h
        sessions[key] = (now, sess)
        lock = session_locks.setdefault(key, asyncio.Lock())
        try:
            orch = get_orch()
        except Exception as e:
            raise HTTPException(503, dict(code="LLM_NOT_CONFIGURED", message=str(getattr(e, "message", e))[:200]))
        async with lock:
            hub = hub_factory(h)
            try:
                async with hub:
                    run = await orch.handle(sess, hub, req.message, req.image_ids)
            except Exception as e:
                raise HTTPException(502, dict(code="TOOLS_UNAVAILABLE", message=f"no se pudo conectar con las tools ({type(e).__name__})"))
        return dict(answer=run.answer, agent=run.agent, stop_reason=run.stop_reason, iterations=run.iterations,
                    tool_calls=[dict(step=t["step"], tool=t["tool"], args=t["args"], ok=t["ok"], code=t["code"], status=t["status"], request_id=t["request_id"],
                                     latency_ms=t["latency_ms"]) for t in run.tool_calls],
                    pending_requests=run.pending_requests,
                    usage=dict(prompt_tokens=run.prompt_tokens, completion_tokens=run.completion_tokens, cost_usd=run.cost_usd, latency_s=run.latency_s))

    # ------------------------------------------------------------------ estado de la linea y aprobaciones
    @app.get("/v1/line")
    def line(ident: auth.Identity = Depends(identity)):
        with db_lock:
            r = conn.execute("SELECT * FROM line_config WHERE line_id=?", (ident.line_id,)).fetchone()
            pend = conn.execute("SELECT count(*) FROM approval_requests WHERE line_id=? AND status='pending'", (ident.line_id,)).fetchone()[0]
        return dict(line_id=ident.line_id, active_model_version=r["active_model_version"] if r else None, confidence_threshold=r["confidence_threshold"] if r else None,
                    line_state=r["line_state"] if r else None, pending_requests=pend)

    @app.get("/v1/approvals")
    def pending(ident: auth.Identity = Depends(identity)):
        with db_lock:
            return approvals.list_pending(ident)

    @app.post("/v1/approvals/{request_id}/approve")
    def approve(request_id: str, ident: auth.Identity = Depends(identity)):
        with db_lock:
            return approvals.approve(ident, request_id)

    @app.post("/v1/approvals/{request_id}/reject")
    def reject(request_id: str, body: RejectRequest, ident: auth.Identity = Depends(identity)):
        with db_lock:
            return approvals.reject(ident, request_id, body.note)

    @app.get("/v1/audit/verify")
    def audit_verify(ident: auth.Identity = Depends(identity)):
        if ident.role not in ("supervisor", "ingeniero_ml"):
            raise HTTPException(403, dict(code="FORBIDDEN", message="su rol no puede consultar la auditoria"))
        with db_lock:
            ok, n, bad = audit.verify_chain(conn)
        return dict(chain_ok=ok, records=n, first_bad_index=bad)

    # ------------------------------------------------------------------ salud y pantalla
    @app.get("/health")
    def health():
        out = dict(status="ok", llm_configured=bool(os.environ.get("AZURE_OPENAI_API_KEY") and not os.environ["AZURE_OPENAI_API_KEY"].startswith(("REEMPLAZAR", "<"))) or state["llm"] is not None)
        try:
            with db_lock:
                conn.execute("SELECT 1").fetchone()
            out["db"] = "ok"
        except Exception:
            out["db"], out["status"] = "error", "degraded"
        try:
            r = httpx.get(cfg.vision_url.rstrip("/") + "/health", timeout=2)
            out["vision"] = "ok" if r.status_code == 200 else f"http {r.status_code}"
        except Exception:
            out["vision"], out["status"] = "unreachable", "degraded"
        return out

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    return app
