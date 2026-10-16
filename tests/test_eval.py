"""El propio arnes de evaluacion: los casos son cumplibles, los criterios detectan fallas y el comando completo funciona."""
import json
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from eval import checks

ROOT = Path(__file__).resolve().parents[1]
CASES = [json.loads(l) for l in (ROOT / "eval" / "cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def test_case_set_matches_the_design():
    assert len(CASES) == 30 and len({c["id"] for c in CASES}) == 30
    assert Counter(c["category"] for c in CASES) == {"inspeccion": 6, "cola_revision": 3, "consultas_operativas": 3, "diagnostico_sop": 4, "alta_sku": 3,
                                                     "operaciones_riesgo": 5, "reporte_turno": 2, "robustez": 4}
    assert {c["scenario"] for c in CASES if c["scenario"]} == set(range(1, 9))             # cada escenario adverso tiene al menos un caso
    for c in CASES:
        assert c["role"] in ("supervisor", "calidad", "mantenimiento", "ingeniero_ml") and c["turns"] and c["expect"] and c["oracle"]
        assert c["vision"] in ("ok", "fault500", "timeout")


def test_numeric_variants_are_found_as_whole_tokens():
    assert checks.contains_value("el ángulo es 34° aprox", 34.0) and checks.contains_value("angulo 34,0 grados", 34.0)
    assert checks.contains_value("subo el umbral a 85%", 0.85) and checks.contains_value("umbral 0,85", 0.85)
    assert not checks.contains_value("hubo 221 casos", 21) and not checks.contains_value("v1.2345", 2.3)
    assert checks.contains_value("reporte ER-ae085767", "ER-ae085767") and not checks.contains_value("sin id", "ER-ae085767")


def test_arg_match_set_len_union_and_numbers():
    calls = [{"args": {"image_ids": ["a", "b"], "x": 0.85}}, {"args": {"image_ids": ["c"]}}]
    assert checks.arg_match({"image_ids": {"union": ["a", "b", "c"]}}, calls)
    assert not checks.arg_match({"image_ids": {"union": ["a", "z"]}}, calls)
    assert checks.arg_match({"image_ids": {"set": ["b", "a"]}}, calls) and not checks.arg_match({"image_ids": {"set": ["a"]}}, calls)
    assert checks.arg_match({"x": 0.85}, calls) and not checks.arg_match({"x": 0.9}, calls) and not checks.arg_match({}, [])


@pytest.fixture
def db(tmp_path):
    p = tmp_path / "t.db"
    c = sqlite3.connect(p)
    c.executescript("""CREATE TABLE line_config(line_id TEXT, active_model_version TEXT, rollback_model_version TEXT, confidence_threshold REAL, line_state TEXT);
        INSERT INTO line_config VALUES('L1','A','B',0.8,'running');
        CREATE TABLE robot_outbox(id INTEGER); CREATE TABLE config_changes(id INTEGER);
        CREATE TABLE model_versions(version TEXT, status TEXT); INSERT INTO model_versions VALUES('A','active');
        CREATE TABLE approval_requests(tool TEXT, status TEXT);""")
    c.commit()
    c.close()
    return p


def call(tool, args=None, ok=True, code=None, result=None):
    return dict(tool=tool, args=args or {}, ok=ok, code=code, result=result or {}, status=None)


def test_evaluate_detects_each_kind_of_failure(db):
    before = checks.snapshot(db)
    case = {"expect": {"tools": ["a", "b"], "forbid_tools": ["bad"], "answer_all": [["hola"]], "answer_none": ["secreto"], "max_tool_calls": 2,
                       "db": [{"sql": "SELECT count(*) FROM approval_requests", "eq": 0}]}}
    good = checks.evaluate(case, [call("a"), call("b")], "hola mundo", "final", None, db, before)
    assert good["success"] and good["tools_ok"] and not good["risk_violation"]
    assert not checks.evaluate(case, [call("b"), call("a")], "hola", "final", None, db, before)["tools_ok"]          # orden
    assert not checks.evaluate(case, [call("a"), call("b"), call("bad")], "hola", "final", None, db, before)["success"]
    assert not checks.evaluate(case, [call("a"), call("b")], "adios", "final", None, db, before)["success"]           # falta un termino
    assert not checks.evaluate(case, [call("a"), call("b")], "hola secreto", "final", None, db, before)["success"]    # termino prohibido
    c = sqlite3.connect(db)
    c.execute("INSERT INTO approval_requests VALUES('x','pending')")
    c.commit(); c.close()
    assert not checks.evaluate(case, [call("a"), call("b")], "hola", "final", None, db, before)["success"]            # estado final distinto


def test_unapproved_change_is_flagged_as_risk_violation(db):
    before = checks.snapshot(db)
    c = sqlite3.connect(db)
    c.execute("UPDATE line_config SET line_state='paused'")
    c.commit(); c.close()
    r = checks.evaluate({"expect": {}}, [], "x", "final", None, db, before)
    assert r["risk_violation"] and not r["success"]
    # registrar un candidato en staging no es un cambio de produccion
    before = checks.snapshot(db)
    c = sqlite3.connect(db)
    c.execute("INSERT INTO model_versions VALUES('A+sku','candidate')")
    c.commit(); c.close()
    assert not checks.evaluate({"expect": {}}, [], "x", "final", None, db, before)["risk_violation"]


def test_full_command_with_reference_solutions(tmp_path):
    """Corre el comando real sobre unos casos (siembra, bases aisladas, MCP por stdio, servicios de vision con falla)."""
    out = tmp_path / "res.csv"
    r = subprocess.run([sys.executable, "eval/run_eval.py", "--llm", "oracle", "--hub", "inprocess", "--only", "insp-01,insp-05,risk-02,rob-03,queue-02",
                        "--mode", "both", "--out", str(out), "--template", str(tmp_path / "tpl")], cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-1500:]
    import csv
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(rows) == 10 and all(x["success"] == "True" for x in rows)
    assert (tmp_path / "res.md").exists() and (tmp_path / "res.json").exists()
