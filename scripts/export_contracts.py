"""Genera los contratos desde el CODIGO (fuente unica de verdad), para que la documentacion no se desfase de lo que corre:

  python scripts/export_contracts.py   ->   docs/TOOLS_CATALOG.md, docs/contracts/tools.json, vision_api.openapi.json, agent_api.openapi.json

tests/test_contracts.py comprueba que los archivos versionados coinciden con el codigo.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from services.common.toolkit import TOOL_VERSION                           # noqa: E402

OUT = ROOT / "docs" / "contracts"

ERRORS = [  # codigo, http, reintentable, significado (derivado de los ToolError del codigo; ver tests/test_contracts.py)
    ("UNAUTHENTICATED", 401, False, "JWT ausente, invalido o vencido"),
    ("FORBIDDEN", 403, False, "el rol del JWT no puede usar la tool"),
    ("FORBIDDEN_LINE", 403, False, "el argumento line_id no coincide con la linea del JWT"),
    ("UNKNOWN_TOOL", 404, False, "la tool no existe en ese servidor (o no esta en el conjunto del agente)"),
    ("VALIDATION_ERROR", 422, False, "los argumentos no cumplen el esquema JSON o una regla de negocio; incluye el campo"),
    ("IMAGE_NOT_FOUND", 404, False, "alguna imagen no existe en la linea del usuario (incluye missing_ids)"),
    ("VISION_TIMEOUT", 504, False, "el servicio de vision no respondio dentro del limite (no se reintenta automaticamente)"),
    ("VISION_UNREACHABLE", 502, True, "no se pudo conectar con el servicio de vision"),
    ("VISION_5XX", 502, True, "el servicio de vision respondio 5xx (la tool ya hizo su reintento controlado)"),
    ("VISION_REJECTED_REQUEST", 422, False, "el servicio de vision rechazo la solicitud (4xx)"),
    ("NOT_INSPECTED", 409, False, "queue_reject solo acepta imagenes ya inspeccionadas"),
    ("NOT_FOUND", 404, False, "caso inexistente en la linea"),
    ("UNKNOWN_CLASS", 422, False, "la clase no esta en el catalogo de SKU"),
    ("CONFLICT", 409, False, "el caso ya tiene otra etiqueta confirmada; no se sobrescribe"),
    ("CASE_NOT_OPEN", 409, False, "el caso no esta abierto"),
    ("TURN_NOT_FOUND", 404, False, "el turno no existe para la linea"),
    ("TURN_NOT_CLOSED", 409, False, "el reporte solo se genera con el turno cerrado"),
    ("NOT_IN_REVIEW_QUEUE", 409, False, "el caso debe estar en la cola de revision de la linea"),
    ("CASE_NOT_ACTIONABLE", 409, False, "el caso no esta en un estado que permita la accion"),
    ("SKU_EXISTS", 409, False, "ya existe un SKU con ese nombre"),
    ("NOT_VALIDATED", 409, False, "el SKU no tiene un lote validado en staging"),
    ("NO_EVIDENCE_AVAILABLE", 501, False, "el MVP solo evalua SKU de demostracion (simulado)"),
    ("EVIDENCE_NOT_FOUND", 404, False, "la evidencia no existe o no corresponde a la version candidata"),
    ("NOT_A_CANDIDATE", 409, False, "la version no esta como candidata"),
    ("EVIDENCE_BELOW_MINIMUMS", 422, False, "la evidencia no alcanza los minimos de promocion; no se crea la solicitud"),
    ("INTERNAL_ERROR", 500, False, "fallo inesperado; el detalle queda solo en la auditoria, no se devuelve al LLM"),
]


def collect():
    from services.mcp_servers import maintenance_server, ml_eng_server, ops_server, vision_server
    out = []
    for mod in (vision_server, ops_server, maintenance_server, ml_eng_server):
        for s in mod.SPECS:
            out.append(dict(name=s.name, server=mod.__name__.rsplit(".", 1)[1], description=s.description,
                            roles=list(s.roles) if s.roles else "todos", effect=("riesgo: solo prepara una solicitud; la decide una persona" if s.risk
                                                                                  else "escritura (idempotente por sesion)" if s.write else "solo lectura"),
                            input_schema=s.schema))
    return out


def render_md(tools):
    L = ["# Catálogo de tools y contratos", "",
         f"Generado por `python scripts/export_contracts.py` desde el código (versión de contrato de las tools: `{TOOL_VERSION}`). No editar a mano.", "",
         "Todas las tools se exponen por **MCP** (JSON-RPC; stdio o streamable-HTTP) y comparten el mismo pipeline: "
         "identidad (JWT) → rol (RBAC) → línea → esquema JSON → idempotencia (escrituras) → tool → auditoría encadenada. "
         "Las respuestas son siempre datos: `{\"ok\": true, ...}` o `{\"ok\": false, \"error\": {code, message, retryable, http_status}}`; nunca excepciones.", "",
         "La **identidad no es un argumento**: `user`, `role` y `line` salen del JWT. Si una tool recibe `line_id` y no coincide con la línea del JWT, responde `FORBIDDEN_LINE`.", "",
         "## Resumen", "", "| Tool | Servidor | Roles | Efecto |", "|---|---|---|---|"]
    for t in tools:
        L.append(f"| `{t['name']}` | {t['server']} | {', '.join(t['roles']) if isinstance(t['roles'], list) else t['roles']} | {t['effect']} |")
    L += ["", "## Contratos de entrada", ""]
    for t in tools:
        L += [f"### `{t['name']}`", "", t["description"], "",
              f"- Servidor: `{t['server']}` · Roles: {', '.join(t['roles']) if isinstance(t['roles'], list) else t['roles']} · Efecto: {t['effect']}", "",
              "```json", json.dumps(t["input_schema"], ensure_ascii=False, indent=2), "```", ""]
    L += ["## Códigos de error", "", "| Código | HTTP | Reintentable | Significado |", "|---|---|---|---|"]
    for c, h, r, m in ERRORS:
        L.append(f"| `{c}` | {h} | {'sí' if r else 'no'} | {m} |")
    L += ["", "Reintentable = el agente puede repetir la misma llamada una vez; el orquestador además corta un segundo intento idéntico (`loop_detected`) "
          "y el máximo de 4 rondas de tools (`max_iterations`).", "",
          "## Contratos de las APIs HTTP", "",
          "- Servicio de visión: `docs/contracts/vision_api.openapi.json` (`POST /v1/predict`, `GET /model`, `GET /health`).",
          "- API del agente y aprobaciones: `docs/contracts/agent_api.openapi.json` (`POST /v1/chat`, `/v1/images`, `/v1/approvals/...`).", ""]
    return "\n".join(L)


def main():
    os.environ.setdefault("JWT_SECRET", "x" * 32)
    os.environ.setdefault("APPROVAL_SECRET", "y" * 32)
    tools = collect()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tools.json").write_text(json.dumps(dict(tool_contract_version=TOOL_VERSION, tools=tools, errors=[dict(code=c, http=h, retryable=r, meaning=m) for c, h, r, m in ERRORS]),
                                               ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (ROOT / "docs" / "TOOLS_CATALOG.md").write_text(render_md(tools), encoding="utf-8")
    from services.vision_api.app import create_app as vision_app
    (OUT / "vision_api.openapi.json").write_text(json.dumps(vision_app().openapi(), ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    import tempfile
    os.environ["OPS_DB_PATH"] = str(Path(tempfile.mkdtemp()) / "x.db")
    from services.agent_api.app import create_app as agent_app
    (OUT / "agent_api.openapi.json").write_text(json.dumps(agent_app(hub_factory=lambda t: None).openapi(), ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(tools)} tools -> docs/TOOLS_CATALOG.md, docs/contracts/*.json")


if __name__ == "__main__":
    main()
