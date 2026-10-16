"""Acceso del orquestador a las tools: dos implementaciones con la misma interfaz.

StdioToolHub  -> arranca los 4 servidores MCP como procesos (protocolo MCP real, JWT por entorno). Es lo que usa la demo y la evaluacion.
InProcessHub  -> llama a ToolServer.invoke en el mismo proceso (pruebas rapidas); pasa por el mismo pipeline de middleware.
El LLM solo ve tool_schemas(); la identidad (JWT) nunca aparece en esos esquemas ni en los argumentos.
"""
import asyncio
import json
import os
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_MODULES = ("vision_server", "ops_server", "maintenance_server", "ml_eng_server")
HTTP_PORTS = {"vision_server": 8101, "ops_server": 8102, "maintenance_server": 8103, "ml_eng_server": 8104}


def _unavailable(name):
    return {"ok": False, "error": {"code": "UNKNOWN_TOOL", "message": f"la tool '{name}' no esta disponible para este agente",
                                   "retryable": False, "http_status": 404}}


def _openai_tool(t):
    return {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["schema"]}}


class _BaseHub:
    def __init__(self):
        self.catalog = {}          # nombre -> {name, description, schema, server}

    def tool_schemas(self, allowed=None):
        return [_openai_tool(t) for n, t in self.catalog.items() if allowed is None or n in allowed]

    async def call(self, name, args, allowed=None):
        """Devuelve (resultado_dict, latencia_ms, servidor). Nunca lanza: los fallos salen como {ok:false,error:{...}}."""
        t0 = time.time()
        if name not in self.catalog or (allowed is not None and name not in allowed):
            return _unavailable(name), round((time.time() - t0) * 1000, 1), None
        srv = self.catalog[name]["server"]
        try:
            out = await self._call(srv, name, args)
        except Exception as e:
            out = {"ok": False, "error": {"code": "TOOL_TRANSPORT_ERROR", "message": f"fallo al invocar la tool ({type(e).__name__})",
                                          "retryable": True, "http_status": 502}}
        return out, round((time.time() - t0) * 1000, 1), srv

    async def aclose(self):
        pass

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *exc):
        await self.aclose()


class InProcessHub(_BaseHub):
    def __init__(self, token, servers=None):
        super().__init__()
        self.token = token
        if servers is None:
            import importlib
            servers = {m: importlib.import_module(f"services.mcp_servers.{m}").make_server() for m in SERVER_MODULES}
        self.servers = servers

    async def start(self):
        for sname, srv in self.servers.items():
            for n, s in srv.specs.items():
                self.catalog[n] = dict(name=n, description=s.description, schema=s.schema, server=sname)

    async def _call(self, srv, name, args):
        return await asyncio.to_thread(self.servers[srv].invoke, name, args, self.token)


class StdioToolHub(_BaseHub):
    def __init__(self, token, env=None, python=None):
        super().__init__()
        self.token, self.env, self.python = token, env or {}, python or sys.executable
        self._stack, self._sessions = None, {}

    async def start(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        self._stack = AsyncExitStack()
        env = {**os.environ, **self.env, "SESSION_JWT": self.token, "PYTHONPATH": str(REPO_ROOT), "PYTHONIOENCODING": "utf-8"}
        try:
            for m in SERVER_MODULES:
                p = StdioServerParameters(command=self.python, args=["-m", f"services.mcp_servers.{m}"], env=env, cwd=str(REPO_ROOT))
                r, w = await self._stack.enter_async_context(stdio_client(p))
                s = await self._stack.enter_async_context(ClientSession(r, w))
                await s.initialize()
                self._sessions[m] = s
                for t in (await s.list_tools()).tools:
                    self.catalog[t.name] = dict(name=t.name, description=t.description or "", schema=t.inputSchema, server=m)
        except BaseException:
            await self.aclose()
            raise

    async def _call(self, srv, name, args):
        res = await self._sessions[srv].call_tool(name, args)
        if getattr(res, "structuredContent", None):
            return res.structuredContent
        return json.loads(res.content[0].text)

    async def aclose(self):
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except BaseException:
                pass
            self._stack, self._sessions = None, {}


class HttpToolHub(_BaseHub):
    """Los 4 servidores MCP como servicios HTTP (despliegue con contenedores). El JWT de la sesion viaja en Authorization: Bearer."""

    def __init__(self, token, urls):
        super().__init__()
        self.token, self.urls = token, urls
        self._stack, self._sessions = None, {}

    @classmethod
    def from_env(cls, token):
        e = os.environ.get
        host = e("MCP_HOST_TEMPLATE", "http://127.0.0.1:{port}/mcp/")
        urls = {m: e("MCP_URL_" + m.replace("_server", "").upper(), host.format(port=HTTP_PORTS[m])) for m in SERVER_MODULES}
        return cls(token, urls)

    async def start(self):
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
        self._stack = AsyncExitStack()
        try:
            for m, url in self.urls.items():
                r, w, _ = await self._stack.enter_async_context(streamablehttp_client(url, headers={"Authorization": f"Bearer {self.token}"}, timeout=20))
                s = await self._stack.enter_async_context(ClientSession(r, w))
                await s.initialize()
                self._sessions[m] = s
                for t in (await s.list_tools()).tools:
                    self.catalog[t.name] = dict(name=t.name, description=t.description or "", schema=t.inputSchema, server=m)
        except BaseException:
            await self.aclose()
            raise

    async def _call(self, srv, name, args):
        res = await self._sessions[srv].call_tool(name, args)
        if getattr(res, "structuredContent", None):
            return res.structuredContent
        return json.loads(res.content[0].text)

    async def aclose(self):
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except BaseException:
                pass
            self._stack, self._sessions = None, {}
