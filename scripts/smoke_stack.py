"""Prueba de humo contra el stack en ejecucion (Docker o run_stack.py):  python scripts/smoke_stack.py [--url http://127.0.0.1:8000] [--chat]

Comprueba salud, login demo, estado de la linea y carga de una imagen; con --chat envia una consulta real al LLM (usa tokens)."""
import argparse
import json
import sys
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--chat", action="store_true")
a = ap.parse_args()


def call(method, path, body=None, token=None):
    req = urllib.request.Request(a.url + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", **({"Authorization": "Bearer " + token} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=190) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("OK   " if cond else "FALLA"), name, detail)


s, h = call("GET", "/health")
check("health", s == 200 and h.get("db") == "ok" and h.get("vision") == "ok", json.dumps(h))
s, t = call("POST", "/auth/token", dict(user_id="ana.supervisor"))
check("login demo", s == 200 and "token" in t)
tok = t.get("token")
s, l = call("GET", "/v1/line", token=tok)
check("estado de la linea", s == 200 and l.get("line_id") == "L1", json.dumps(l))
check("sin token -> 401", call("GET", "/v1/line")[0] == 401)
s, ap_ = call("GET", "/v1/approvals", token=tok)
check("lista de aprobaciones", s == 200, f"{len(ap_)} pendientes")
if a.chat:
    s, r = call("POST", "/v1/chat", dict(message="Muestrame la cola de revision"), tok)
    check("chat con el LLM real", s == 200 and r.get("answer"), f"agente={r.get('agent')} tools={[c['tool'] for c in r.get('tool_calls', [])]} tokens={r.get('usage')}")
sys.exit(0 if ok else 1)
