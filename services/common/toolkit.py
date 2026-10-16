"""Capa comun de los servidores MCP: identidad -> RBAC -> linea -> esquema JSON -> idempotencia -> tool -> auditoria.

Cada servidor (vision, operaciones, mantenimiento, ingenieria ML) declara sus tools como ToolSpec y reutiliza este pipeline.
Los errores salen como datos estructurados {ok:false, error:{code,message,retryable,http_status}}, nunca como excepciones.
"""
import asyncio
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import jsonschema

from . import audit, auth, db
from .approvals import ApprovalService
from .config import Config

TOOL_VERSION = "1.0"


class ToolError(Exception):
    def __init__(self, code, message, retryable=False, status=400, **extra):
        super().__init__(message)
        self.code, self.message, self.retryable, self.status, self.extra = code, message, retryable, status, extra


@dataclass
class ToolSpec:
    name: str
    description: str
    schema: dict
    handler: Callable
    roles: tuple | None = None          # None = todos los roles
    write: bool = False
    risk: bool = False


@dataclass
class ToolCtx:
    ident: auth.Identity
    conn: object
    cfg: Config
    approvals: ApprovalService


def _canon(o):
    return json.dumps(o, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def err(code, message, retryable=False, status=400, **extra):
    return {"ok": False, "error": {"code": code, "message": message, "retryable": retryable, "http_status": status, **extra}}


class ToolServer:
    def __init__(self, name, specs, cfg=None, conn=None):
        self.name = name
        self.specs = {s.name: s for s in specs}
        self.cfg = cfg or Config.load()
        self.conn = conn or db.init_schema(db.connect(self.cfg.db_path))
        self.approvals = ApprovalService(self.conn, self.cfg)
        self._lock = threading.Lock()
        from . import effects                                     # ejecutores de las operaciones de riesgo
        self.approvals.executors = effects.EXECUTORS

    # ------------------------------------------------------------------ pipeline (sincrono; lo usan MCP, pruebas y evaluacion)
    def invoke(self, tool, args, token=None, identity=None):
        t0 = time.time()
        args = args if isinstance(args, dict) else {}
        ident, out = identity, None
        rec = dict(server=self.name, tool=tool, tool_version=TOOL_VERSION, args=_redact(args))
        try:
            try:
                ident = ident or auth.verify_token(self.cfg.jwt_secret, token)
            except auth.AuthError as e:
                out = err("UNAUTHENTICATED", e.message, status=401)
                return out
            rec.update(user=ident.user_id, role=ident.role, line=ident.line_id, session=ident.session_id)
            spec = self.specs.get(tool)
            if not spec:
                out = err("UNKNOWN_TOOL", f"la tool '{tool}' no existe en este servidor", status=404)
                return out
            if spec.roles is not None and ident.role not in spec.roles:
                out = err("FORBIDDEN", f"el rol '{ident.role}' no puede usar {tool}", status=403)
                return out
            if "line_id" in args and args["line_id"] != ident.line_id:       # la linea viene del JWT; un valor distinto se deniega
                out = err("FORBIDDEN_LINE", "solo puede consultar la linea de su sesion", status=403)
                return out
            try:
                jsonschema.Draft202012Validator(spec.schema).validate(args)
            except jsonschema.ValidationError as e:
                field_ = ".".join(str(p) for p in e.absolute_path) or (e.validator_value[0] if e.validator == "required" else "")
                out = err("VALIDATION_ERROR", e.message[:300], status=422, field=field_)
                return out
            ctx = ToolCtx(ident, self.conn, self.cfg, self.approvals)
            ikey = None
            if spec.write and not spec.risk:
                ikey = hashlib.sha256(_canon([tool, args, ident.session_id, ident.line_id]).encode()).hexdigest()
                row = self.conn.execute("SELECT result_json FROM idempotency WHERE key=?", (ikey,)).fetchone()
                if row:
                    out = json.loads(row["result_json"])
                    out["idempotent_replay"] = True
                    rec["idempotent_replay"] = True
                    return out
            try:
                data = spec.handler(ctx, args) or {}
                rec.update(data.pop("_audit", {}))
                out = {"ok": True, **data}
                if ikey:
                    self.conn.execute("INSERT OR IGNORE INTO idempotency(key,tool,result_json,created_at) VALUES(?,?,?,?)",
                                      (ikey, tool, _canon(out), audit.now_iso()))
            except ToolError as e:
                rec.update(e.extra.pop("_audit", {}))
                out = err(e.code, e.message, e.retryable, e.status, **e.extra)
            except Exception as e:                                              # fallo inesperado: nunca se filtra el detalle al LLM
                out = err("INTERNAL_ERROR", "error interno de la tool", False, 500)
                rec["exception"] = f"{type(e).__name__}: {str(e)[:200]}"
            return out
        finally:
            if out is not None:
                rec.update(ok=out.get("ok", False), code=None if out.get("ok") else out["error"]["code"],
                           latency_ms=round((time.time() - t0) * 1000, 1))
                try:
                    audit.append(self.conn, rec)
                except Exception:                                               # la auditoria no debe tumbar la respuesta, pero se deja constancia
                    out.setdefault("warnings", []).append("AUDIT_WRITE_FAILED")

    # ------------------------------------------------------------------ adaptador MCP
    def build_mcp(self):
        from mcp import types
        from mcp.server.lowlevel import Server
        server = Server(self.name)

        @server.list_tools()
        async def _list():
            return [types.Tool(name=s.name, description=s.description, inputSchema=s.schema) for s in self.specs.values()]

        @server.call_tool(validate_input=False)          # la validacion la hace invoke() para devolver VALIDATION_ERROR estructurado
        async def _call(name, arguments):
            token = _token_from_request(server) or os.environ.get("SESSION_JWT")
            def locked():
                with self._lock:                      # una llamada a la vez por servidor: la conexion SQLite no se comparte entre hilos
                    return self.invoke(name, arguments or {}, token)
            return await asyncio.to_thread(locked)

        return server

    def run(self):
        """stdio (por defecto) o http (MCP_TRANSPORT=streamable-http, MCP_HOST, MCP_PORT; el JWT va en Authorization: Bearer)."""
        server = self.build_mcp()
        transport = os.environ.get("MCP_TRANSPORT", "stdio")
        if transport == "stdio":
            from mcp.server.stdio import stdio_server

            async def main():
                async with stdio_server() as (r, w):
                    await server.run(r, w, server.create_initialization_options())
            asyncio.run(main())
        elif transport == "streamable-http":
            self._run_http(server)
        else:
            raise SystemExit(f"MCP_TRANSPORT desconocido: {transport} (use stdio o streamable-http)")

    def _run_http(self, server):
        """MCP por HTTP (stateless): cada llamada trae su JWT en Authorization: Bearer. /health para el orquestador de contenedores."""
        import contextlib

        import uvicorn
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Mount, Route

        manager = StreamableHTTPSessionManager(app=server, json_response=True, stateless=True)

        async def mcp_app(scope, receive, send):
            await manager.handle_request(scope, receive, send)

        async def health(request):
            return JSONResponse({"status": "ok", "server": self.name, "tools": sorted(self.specs)})

        @contextlib.asynccontextmanager
        async def lifespan(app):
            async with manager.run():
                yield

        app = Starlette(routes=[Route("/health", health), Mount("/mcp", app=mcp_app)], lifespan=lifespan)
        uvicorn.run(app, host=os.environ.get("MCP_HOST", "127.0.0.1"), port=int(os.environ.get("MCP_PORT", "8100")), log_level="warning")


def _token_from_request(server):
    try:
        req = server.request_context.request
        h = req.headers.get("authorization", "") if req is not None else ""
        return h[7:] if h.lower().startswith("bearer ") else None
    except Exception:
        return None


def _redact(args):
    out = {}
    for k, v in args.items():
        out[k] = v[:200] + "..." if isinstance(v, str) and len(v) > 200 else v
    return out
