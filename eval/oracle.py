"""LLM de referencia para la AUTOPRUEBA del arnes: ejecuta la solucion de referencia que lleva cada caso (campo `oracle`).

Sirve para comprobar que los casos son cumplibles y que el arnes (bases aisladas, MCP real, servicios de vision con fallas, CSV) funciona.
Sus tokens, costos y latencias son ficticios: NUNCA se reportan como resultados de evaluacion."""
import json
import re

from services.agent.testing import ScriptedLLM

REF = re.compile(r"\{\{([^}]+)\}\}")


def _dig(obj, path):
    for part in path.split("."):
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


def _results(messages):
    out = {}
    for m in messages:
        if m.get("role") == "tool":
            try:
                out[m["name"]] = json.loads(m["content"])
            except ValueError:
                pass
    return out


def _fill(v, res):
    if isinstance(v, str):
        m = REF.fullmatch(v.strip())
        if m:
            tool, _, path = m.group(1).partition(".")
            return _dig(res[tool], path)
        return REF.sub(lambda mm: str(_dig(res[mm.group(1).partition(".")[0]], mm.group(1).partition(".")[2])), v)
    if isinstance(v, list):
        return [_fill(x, res) for x in v]
    if isinstance(v, dict):
        return {k: _fill(x, res) for k, x in v.items()}
    return v


def oracle_llm(steps, mode):
    seq = []
    for s in steps:
        if "route" in s and mode == "single":
            if "single_text" in s:                       # el agente unico no tiene enrutador: responde directamente
                seq.append({"text": s["single_text"]})
            continue
        seq.append(s)

    def policy(messages, tools, n):
        if n >= len(seq):
            return {"text": "fin"}
        s, res = seq[n], _results(messages)
        if "route" in s:
            return {"text": s["route"]}
        if "tools" in s:
            return {"tools": [(name, _fill(args, res)) for name, args in s["tools"]]}
        return {"text": _fill(s["text"], res)}
    return ScriptedLLM(policy)
