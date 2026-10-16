"""Los contratos versionados (docs/) deben coincidir con el codigo, y el catalogo de errores debe cubrir todos los codigos que el codigo emite."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_error_catalog_covers_every_code_in_source():
    import export_contracts as ec
    emitted = set()
    for f in list((ROOT / "services").rglob("*.py")):
        emitted |= set(re.findall(r'(?:ToolError|err)\("([A-Z][A-Z_0-9]+)"', f.read_text(encoding="utf-8")))
    assert emitted - {c for c, *_ in ec.ERRORS} == set(), "codigos de error sin documentar"


def test_committed_contracts_match_code(world):
    import export_contracts as ec
    cur = json.loads((ROOT / "docs" / "contracts" / "tools.json").read_text(encoding="utf-8"))
    now = ec.collect()
    assert [t["name"] for t in cur["tools"]] == sorted([t["name"] for t in now]) or {t["name"] for t in cur["tools"]} == {t["name"] for t in now}
    by = {t["name"]: t for t in cur["tools"]}
    for t in now:
        assert by[t["name"]]["input_schema"] == t["input_schema"], f"{t['name']}: el esquema versionado no coincide; ejecute scripts/export_contracts.py"
        assert by[t["name"]]["roles"] == t["roles"]
    assert (ROOT / "docs" / "TOOLS_CATALOG.md").read_text(encoding="utf-8") == ec.render_md(now)
