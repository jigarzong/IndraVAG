"""Cargador minimo de .env para los scripts (no pisa variables ya definidas en el entorno; no imprime valores)."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_env(path=None):
    p = Path(path) if path else REPO_ROOT / ".env"
    if not p.exists():
        return False
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        if k and k not in os.environ:
            os.environ[k] = v
    return True
