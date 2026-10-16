"""Crea .env a partir de .env.example con secretos aleatorios (no pisa un .env existente)."""
import secrets
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
dst = root / ".env"
if dst.exists():
    sys.exit(".env ya existe; no se modifica")
text = (root / ".env.example").read_text(encoding="utf-8")
out = []
for line in text.splitlines():
    if line.startswith(("JWT_SECRET=", "APPROVAL_SECRET=")):
        line = line.split("=")[0] + "=" + secrets.token_hex(32)
    out.append(line)
dst.write_text("\n".join(out) + "\n", encoding="utf-8")
print("creado .env (completa las variables AZURE_OPENAI_* cuando tengas el despliegue)")
