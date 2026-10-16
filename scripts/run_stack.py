"""Levanta TODO el MVP sin Docker, con un comando:  python scripts/run_stack.py

  vision (8001) + 4 servidores MCP por HTTP (8101-8104) + API del agente y pantalla (8000)

Lee .env, siembra los datos de la demo si faltan (scripts/bootstrap.py), arranca los procesos, espera sus /health y, con Ctrl+C,
los detiene. Los registros quedan en logs/. Para Docker use `docker compose up --build`.
"""
import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.common.envfile import load_env                               # noqa: E402

MCP = (("vision_server", 8101), ("ops_server", 8102), ("maintenance_server", 8103), ("ml_eng_server", 8104))


def healthy(url):
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-bootstrap", action="store_true")
    ap.add_argument("--fault-injection", action="store_true", help="habilita VISION_FAULT_INJECTION para la demo de fallas del servicio de vision")
    a = ap.parse_args()
    load_env()
    if len(os.environ.get("JWT_SECRET", "")) < 32 or len(os.environ.get("APPROVAL_SECRET", "")) < 32:
        sys.exit("Faltan JWT_SECRET/APPROVAL_SECRET (>= 32 caracteres). Ejecute: python scripts/init_env.py")
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    py = sys.executable
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8", "VISION_SERVICE_URL": "http://127.0.0.1:8001",
           "AGENT_HUB": "http", "MCP_HOST": "127.0.0.1"}
    for k in ("MCP_URL_VISION", "MCP_URL_OPS", "MCP_URL_MAINTENANCE", "MCP_URL_ML_ENG"):
        env.pop(k, None)
    if a.fault_injection:
        env["VISION_FAULT_INJECTION"] = "1"
    if not a.no_bootstrap:
        r = subprocess.run([py, str(ROOT / "scripts" / "bootstrap.py")], cwd=ROOT, env=env)
        if r.returncode:
            sys.exit("bootstrap fallo")
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    procs, checks = [], []

    def launch(name, args, extra=None):
        f = open(logs / f"{name}.log", "wb")
        procs.append((name, subprocess.Popen(args, cwd=ROOT, env={**env, **(extra or {})}, stdout=f, stderr=subprocess.STDOUT)))

    launch("vision", [py, "-m", "uvicorn", "services.vision_api.app:app", "--host", "127.0.0.1", "--port", "8001"])
    checks.append(("vision", "http://127.0.0.1:8001/health"))
    for m, port in MCP:
        launch(m, [py, "-m", f"services.mcp_servers.{m}"], dict(MCP_TRANSPORT="streamable-http", MCP_PORT=str(port)))
        checks.append((m, f"http://127.0.0.1:{port}/health"))
    launch("agent_api", [py, "-m", "uvicorn", "--factory", "services.agent_api.app:create_app", "--host", a.host, "--port", str(a.port)])
    checks.append(("agent_api", f"http://127.0.0.1:{a.port}/health"))
    try:
        t0, pending = time.time(), dict(checks)
        while pending and time.time() - t0 < 90:
            for n, p in procs:
                if p.poll() is not None:
                    raise SystemExit(f"{n} termino con codigo {p.returncode}; vea logs/{n}.log")
            pending = {n: u for n, u in pending.items() if not healthy(u)}
            time.sleep(0.5)
        if pending:
            raise SystemExit("no respondieron a tiempo: " + ", ".join(pending) + " (vea logs/)")
        print(f"\nListo. Pantalla del asistente:  http://127.0.0.1:{a.port}/\n  API: http://127.0.0.1:{a.port}/docs   |   registros en logs/   |   Ctrl+C detiene todo")
        while True:
            for n, p in procs:
                if p.poll() is not None:
                    raise SystemExit(f"{n} se detuvo (codigo {p.returncode}); vea logs/{n}.log")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\ndeteniendo...")
    finally:
        for _, p in procs:
            if p.poll() is None:
                p.terminate()
        for _, p in procs:
            try:
                p.wait(timeout=8)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
