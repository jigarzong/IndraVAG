"""Evaluacion reproducible del asistente: un comando, 30 casos, una fila por caso en CSV.

  python eval/run_eval.py                          # LLM real (lee .env), agente multiple, escribe eval/results.csv
  python eval/run_eval.py --mode both              # tambien el agente unico con las 15 tools (comparacion de arquitectura)
  python eval/run_eval.py --only insp-01,risk-02   # casos sueltos
  python eval/run_eval.py --llm oracle             # AUTOPRUEBA del arnes con la solucion de referencia (no son resultados)

Cada caso corre con su propia copia de la base y de las imagenes sembradas (eval/.template), identidad por JWT y los 4 servidores MCP
reales por stdio. El servicio de vision es real; los casos de falla usan instancias con inyeccion de 5xx y de retraso.
Metricas: exito, seleccion de tools (orden y argumentos), seguridad, tokens, costo con config/pricing.json y latencia.
"""
import argparse
import asyncio
import csv
import json
import math
import os
import platform
import secrets
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from eval.checks import evaluate, snapshot                                  # noqa: E402
from services.agent.hub import InProcessHub, StdioToolHub                  # noqa: E402
from services.agent.orchestrator import Orchestrator, Session              # noqa: E402
from services.agent.pricing import load_rates                              # noqa: E402
from services.common import auth                                           # noqa: E402
from services.common.envfile import load_env                               # noqa: E402

COLUMNS = ["case_id", "category", "scenario", "mode", "run", "success", "tools_ok", "args_ok", "selection_ok", "risk_violation", "router_ok",
           "agent", "stop_reason", "iterations", "n_tool_calls", "tools_called", "prompt_tokens", "completion_tokens", "cached_tokens",
           "reasoning_tokens", "llm_calls", "cost_usd", "latency_s", "failures", "answer"]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class VisionServices:
    """Servicio de vision real (normal) y dos instancias con falla inyectada, levantadas solo si algun caso las necesita."""
    FAULTS = {"ok": {}, "fault500": {"VISION_FAULT_INJECTION": "1", "VISION_FAULT_MODE": "500"},
              "timeout": {"VISION_FAULT_INJECTION": "1", "VISION_FAULT_MODE": "delay:8"}}

    def __init__(self, env):
        self.env, self.procs, self.urls = env, {}, {}

    def url(self, kind):
        if kind not in self.urls:
            port = free_port()
            e = {**os.environ, **self.env, **self.FAULTS[kind], "PYTHONPATH": f"{ROOT}{os.pathsep}{ROOT / 'src'}"}
            p = subprocess.Popen([sys.executable, "-m", "uvicorn", "services.vision_api.app:app", "--port", str(port), "--log-level", "warning"],
                                 cwd=str(ROOT), env=e, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.procs[kind], self.urls[kind] = p, f"http://127.0.0.1:{port}"
            t0 = time.time()
            while time.time() - t0 < 90:
                try:
                    urllib.request.urlopen(self.urls[kind] + "/health", timeout=2).read()
                    break
                except Exception:
                    if p.poll() is not None:
                        raise RuntimeError(f"el servicio de vision ({kind}) termino al arrancar")
                    time.sleep(0.5)
            else:
                raise RuntimeError(f"el servicio de vision ({kind}) no respondio en 90 s")
        return self.urls[kind]

    def close(self):
        for p in self.procs.values():
            p.terminate()
        for p in self.procs.values():
            try:
                p.wait(10)
            except Exception:
                p.kill()


def ensure_template(tdir, secrets_env, reseed):
    db = tdir / "ops.db"
    ready = tdir / ".ready"                      # solo se escribe cuando el sembrado termino bien: una base a medias no cuenta como plantilla
    if reseed or not ready.exists():
        cache = ROOT / "data" / "cache" / "coil100.npz"
        if not cache.exists():
            raise SystemExit("Falta data/cache/coil100.npz (cache de COIL-100), necesario para sembrar las imagenes de demostracion.\n"
                             "Crealo una vez desde la raiz del proyecto (PowerShell):\n"
                             "  $env:INDRA_DATA='data/coil-100'; $env:CACHE_ONLY='1'; python src/prepare_data.py\n"
                             "y vuelve a ejecutar la evaluacion.")
        print(f"sembrando datos simulados en {tdir} ...", flush=True)
        shutil.rmtree(tdir, ignore_errors=True)
        tdir.mkdir(parents=True)
        env = {**os.environ, **secrets_env, "OPS_DB_PATH": str(db), "IMAGE_STORE_DIR": str(tdir / "uploads")}
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "seed_demo.py"), "--reset"], cwd=str(ROOT), env=env, capture_output=True, text=True)
        if r.returncode != 0:
            shutil.rmtree(tdir, ignore_errors=True)
            raise SystemExit("fallo el sembrado:\n" + r.stdout[-1500:] + r.stderr[-1500:])
        if not (tdir / "uploads" / "img_grasp_01.png").exists():
            shutil.rmtree(tdir, ignore_errors=True)
            raise SystemExit("el sembrado termino pero no genero las imagenes de demostracion (revisa IMAGE_STORE_DIR)")
        c = sqlite3.connect(db)
        c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        c.close()
        ready.write_text("ok")
    return db


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    return xs[max(0, math.ceil(p / 100 * len(xs)) - 1)] if xs else None


async def run_case(case, mode, run_i, a, ctx):
    tmp = Path(tempfile.mkdtemp(prefix="evalcase_"))
    try:
        shutil.copy(ctx["template"] / "ops.db", tmp / "ops.db")
        shutil.copytree(ctx["template"] / "uploads", tmp / "uploads")
        env = {"OPS_DB_PATH": str(tmp / "ops.db"), "IMAGE_STORE_DIR": str(tmp / "uploads"), "VISION_SERVICE_URL": ctx["vision"].url(case.get("vision", "ok")),
               "JWT_SECRET": ctx["secrets"]["JWT_SECRET"], "APPROVAL_SECRET": ctx["secrets"]["APPROVAL_SECRET"], "DEMO_TODAY": ctx["today"]}
        sid = f"eval-{case['id']}-{mode}-{run_i}"
        token = auth.issue_token(env["JWT_SECRET"], case["user"], case["role"], case["line"], sid)
        sess = Session(case["user"], case["role"], case["line"], sid, token)
        if a.hub == "inprocess":
            os.environ.update(env)
            hub = InProcessHub(token)
        else:
            hub = StdioToolHub(token, env=env)
        before = snapshot(tmp / "ops.db")
        llm = ctx["llm_factory"](case, mode)
        orch = Orchestrator(llm, mode=mode, rates=ctx["rates"], today=ctx["today"])
        calls, answer, agg = [], "", dict(prompt_tokens=0, completion_tokens=0, cached_tokens=0, reasoning_tokens=0, llm_calls=0, cost=0.0, latency=0.0, iterations=0)
        first_agent, stop, costs_known = None, None, ctx["rates"] is not None
        async with hub:
            for t in case["turns"]:
                r = await orch.handle(sess, hub, t["text"], t.get("image_ids"))
                first_agent = first_agent or r.agent
                calls += r.tool_calls
                answer, stop = r.answer, r.stop_reason
                for k in ("prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens"):
                    agg[k] += getattr(r, k)
                agg["llm_calls"] += len(r.llm_calls)
                agg["iterations"] += r.iterations
                agg["latency"] += r.latency_s
                agg["cost"] += r.cost_usd or 0.0
        ev = evaluate(case, calls, answer, stop, first_agent, tmp / "ops.db", before)
        return dict(case_id=case["id"], category=case["category"], scenario=case.get("scenario") or "", mode=mode, run=run_i, success=ev["success"],
                    tools_ok=ev["tools_ok"], args_ok=ev["args_ok"], selection_ok=ev["tools_ok"] and ev["args_ok"], risk_violation=ev["risk_violation"],
                    router_ok="" if (ev["router_ok"] is None or mode == "single") else ev["router_ok"], agent=first_agent, stop_reason=stop, iterations=agg["iterations"],
                    n_tool_calls=len(calls), tools_called=">".join(c["tool"] for c in calls), prompt_tokens=agg["prompt_tokens"],
                    completion_tokens=agg["completion_tokens"], cached_tokens=agg["cached_tokens"], reasoning_tokens=agg["reasoning_tokens"],
                    llm_calls=agg["llm_calls"], cost_usd=round(agg["cost"], 6) if costs_known else "", latency_s=round(agg["latency"], 2),
                    failures=" | ".join(ev["failures"]), answer=answer.replace("\n", " ")[:600]), calls
    except Exception as e:                                          # un fallo de infraestructura se registra, no tumba la corrida
        return dict(case_id=case["id"], category=case["category"], scenario=case.get("scenario") or "", mode=mode, run=run_i, success=False,
                    tools_ok=False, args_ok=False, selection_ok=False, risk_violation=False, router_ok="", agent="", stop_reason="infra_error",
                    iterations=0, n_tool_calls=0, tools_called="", prompt_tokens=0, completion_tokens=0, cached_tokens=0, reasoning_tokens=0, llm_calls=0,
                    cost_usd="", latency_s=0, failures=f"INFRA: {type(e).__name__}: {str(e)[:300]}", answer=""), []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def print_trace(case, row, calls):
    """Traza legible de un caso (para la demo de escenarios adversos): que se pidio, que tools se ejecutaron y que respondio el asistente."""
    print("-" * 100)
    print(f"{case['id']} | escenario {case.get('scenario')} | usuario {case['user']} ({case['role']}, linea {case['line']}) | {case.get('note', '')}")
    for t in case["turns"]:
        print(f"  USUARIO: {t['text']}")
    print(f"  ENRUTADO A: {row['agent']}   parada: {row['stop_reason']}   iteraciones: {row['iterations']}")
    for c in calls:
        res = c.get("result") or {}
        out = f"ok status={c['status']} request_id={c['request_id']}" if c["ok"] and c["status"] else ("ok" if c["ok"] else f"ERROR {c['code']}: {c.get('message')}")
        print(f"  [{c['step']}] {c['tool']}({json.dumps(c['args'], ensure_ascii=False)[:140]}) -> {out}")
    print(f"  RESPUESTA: {row['answer']}")
    print(f"  VEREDICTO: {'cumple' if row['success'] else 'NO cumple'} | riesgo sin aprobar: {'SI' if row['risk_violation'] else 'no'} | "
          f"tokens {row['prompt_tokens']}+{row['completion_tokens']} | {row['latency_s']} s" + (f" | {row['failures'][:300]}" if row["failures"] else ""))


def rate(rows, key):
    return f"{100 * sum(1 for r in rows if r[key] is True) / len(rows):.1f}%" if rows else "-"


def summarize(rows, title):
    L = [f"### {title}", ""]
    for mode in sorted({r["mode"] for r in rows}):
        R = [r for r in rows if r["mode"] == mode]
        adv = [r for r in R if r["scenario"] != ""]
        routed = [r for r in R if r["router_ok"] != ""]
        toks = [r["prompt_tokens"] + r["completion_tokens"] for r in R]
        costs = [r["cost_usd"] for r in R if r["cost_usd"] != ""]
        lat = [r["latency_s"] for r in R]
        L += [f"**Modo `{mode}`** ({len(R)} ejecuciones)", "",
              "| Métrica | Valor |", "|---|---|",
              f"| Éxito de la tarea | {rate(R, 'success')} ({sum(1 for r in R if r['success'])}/{len(R)}) |",
              f"| Selección de tools (orden y argumentos) | {rate(R, 'selection_ok')} |",
              f"| Seguridad: casos adversos sin acción de riesgo sin aprobar | {100 * sum(1 for r in adv if not r['risk_violation']) / len(adv):.1f}% ({sum(1 for r in adv if not r['risk_violation'])}/{len(adv)}) |" if adv else "| Seguridad | - |",
              f"| Acierto del enrutador | {rate(routed, 'router_ok') if routed else 'n/a (agente único)'} |",
              f"| Tokens por caso (promedio / p95) | {sum(toks) / len(toks):.0f} / {pct(toks, 95)} |",
              f"| Tokens entrada / salida (promedio) | {sum(r['prompt_tokens'] for r in R) / len(R):.0f} / {sum(r['completion_tokens'] for r in R) / len(R):.0f} |",
              f"| Costo por caso USD (promedio / total) | {sum(costs) / len(costs):.6f} / {sum(costs):.4f} |" if costs else "| Costo | sin tarifa configurada |",
              f"| Latencia por caso s (p50 / p95) | {pct(lat, 50)} / {pct(lat, 95)} |", ""]
        L += ["| Categoría | Casos | Éxito | Selección |", "|---|---|---|---|"]
        for c in sorted({r["category"] for r in R}):
            RC = [r for r in R if r["category"] == c]
            L.append(f"| {c} | {len(RC)} | {rate(RC, 'success')} | {rate(RC, 'selection_ok')} |")
        bad = [r for r in R if not r["success"] or not r["selection_ok"]]
        if bad:
            L += ["", "Casos con fallas:", ""]
            for r in bad:
                L.append(f"- `{r['case_id']}` ({'éxito' if r['success'] else 'FALLA'}, selección {'ok' if r['selection_ok'] else 'FALLA'}): {r['failures'][:400]}")
        L.append("")
    return "\n".join(L)


async def amain(a):
    load_env()
    sec = {k: os.environ.get(k) or secrets.token_hex(32) for k in ("JWT_SECRET", "APPROVAL_SECRET")}
    os.environ.update(sec)
    cases = [json.loads(l) for l in (ROOT / a.cases).read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.only:
        want = set(a.only.split(","))
        cases = [c for c in cases if c["id"] in want]
    if a.scenarios:
        cases = [c for c in cases if c.get("scenario")]
    if not cases:
        raise SystemExit("ningun caso seleccionado")
    template = ensure_template(ROOT / a.template, sec, a.reseed).parent
    rates = load_rates(os.environ.get("PRICING_MODEL_KEY"))
    if a.llm == "real":
        from services.agent.llm import LLMError, SKChatClient
        try:
            real = SKChatClient.from_env()
        except LLMError as e:
            raise SystemExit(f"LLM no configurado: {e.message} (revisa .env y corre scripts/check_llm.py)")
        factory = lambda case, mode: real
        out = a.out
    else:
        from eval.oracle import oracle_llm
        factory = lambda case, mode: oracle_llm(case["oracle"], mode)
        out = a.out if a.out != "eval/results.csv" else "eval/selftest_results.csv"
        print("*** AUTOPRUEBA con LLM de referencia: tokens, costo y latencia NO son resultados de evaluación ***", flush=True)
    vision = VisionServices({**sec, "OPS_DB_PATH": str(template / "ops.db"), "IMAGE_STORE_DIR": str(template / "uploads")})
    ctx = dict(template=template, vision=vision, secrets=sec, rates=rates, today=os.environ.get("DEMO_TODAY", "2026-10-15"), llm_factory=factory)
    modes = ["multi", "single"] if a.mode == "both" else [a.mode]
    rows, details = [], []
    outp = ROOT / out
    outp.parent.mkdir(parents=True, exist_ok=True)
    try:
        for mode in modes:
            for run_i in range(1, a.repeat + 1):
                for case in cases:
                    row, calls = await run_case(case, mode, run_i, a, ctx)
                    if row["stop_reason"] == "infra_error" and not any(r["stop_reason"] != "infra_error" for r in rows):
                        raise SystemExit(f"Fallo de infraestructura en el primer caso ({case['id']}); no se continua para no generar resultados invalidos:\n{row['failures']}")
                    rows.append(row)
                    details.append(dict(case_id=case["id"], mode=mode, run=run_i, answer=row["answer"], failures=row["failures"],
                                        tool_calls=[{k: (str(v)[:1200] if k == "result" else v) for k, v in c.items()} for c in calls]))
                    if a.trace:
                        print_trace(case, row, calls)
                    flag = "OK  " if row["success"] else "FALLA"
                    print(f"[{flag}] {mode:6} {case['id']:9} tools={row['tools_called'] or '-':60.60} tok={row['prompt_tokens']}+{row['completion_tokens']} "
                          f"{row['latency_s']}s" + (f"  <- {row['failures'][:160]}" if not row["success"] else ""), flush=True)
                    with outp.open("w", newline="", encoding="utf-8") as f:
                        w = csv.DictWriter(f, fieldnames=COLUMNS)
                        w.writeheader()
                        w.writerows(rows)
    finally:
        vision.close()
    meta = dict(fecha=time.strftime("%Y-%m-%d %H:%M:%S"), llm=a.llm, deployment=(os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME") or os.environ.get("AZURE_OPENAI_DEPLOYMENT")), pricing_key=os.environ.get("PRICING_MODEL_KEY"),
                rates=rates, temperature=os.environ.get("LLM_TEMPERATURE") or "no fijada", max_iterations=os.environ.get("MAX_AGENT_ITERATIONS", "4"),
                casos=len(cases), repeticiones=a.repeat, python=platform.python_version(), hub=a.hub)
    (outp.with_suffix(".json")).write_text(json.dumps(dict(meta=meta, details=details), ensure_ascii=False, indent=1), encoding="utf-8")
    title = "AUTOPRUEBA (LLM de referencia; no son resultados)" if a.llm != "real" else f"Resultados: {meta['deployment']} - {meta['fecha']}"
    md = summarize(rows, title) + "\n\n```json\n" + json.dumps(meta, ensure_ascii=False, indent=1) + "\n```\n"
    outp.with_suffix(".md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"escrito: {out}, {Path(out).with_suffix('.md')}, {Path(out).with_suffix('.json')}")
    return 0 if all(r["success"] for r in rows) else 1


def main():
    for stream in (sys.stdout, sys.stderr):                  # consolas de Windows con codificacion distinta de UTF-8
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default="eval/cases.jsonl")
    ap.add_argument("--mode", choices=["multi", "single", "both"], default="multi")
    ap.add_argument("--llm", choices=["real", "oracle"], default="real")
    ap.add_argument("--hub", choices=["stdio", "inprocess"], default="stdio")
    ap.add_argument("--out", default="eval/results.csv")
    ap.add_argument("--only", default="")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--template", default="eval/.template")
    ap.add_argument("--reseed", action="store_true")
    ap.add_argument("--scenarios", action="store_true", help="solo los casos de los 8 escenarios adversos")
    ap.add_argument("--trace", action="store_true", help="imprime la traza legible de cada caso")
    a = ap.parse_args()
    sys.exit(asyncio.run(amain(a)))


if __name__ == "__main__":
    main()
