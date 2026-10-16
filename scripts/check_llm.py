"""Prueba de conexion al LLM (Azure AI Foundry, API v1) con el mismo cliente que usa el agente. Lee .env y NO imprime la clave.

Uso:  python scripts/check_llm.py
Comprueba: autenticacion y despliegue, conteo de tokens, llamadas a tools, temperatura 0 y la tarifa configurada.
"""
import asyncio
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.agent.llm import LLMError, SKChatClient                      # noqa: E402
from services.agent.pricing import cost_usd, load_rates                    # noqa: E402
from services.common.envfile import load_env                               # noqa: E402

TOOL = [{"type": "function", "function": {
    "name": "inspect_image", "description": "Inspecciona de 1 a 5 imagenes por id y devuelve la decision.",
    "parameters": {"type": "object", "additionalProperties": False, "required": ["image_ids"],
                   "properties": {"image_ids": {"type": "array", "minItems": 1, "maxItems": 5, "items": {"type": "string"}}}}}}]


def line(ok, text):
    print(("[OK]   " if ok is True else "[FALLA] " if ok is False else "[INFO] ") + text)


async def probe(name, **kw):
    cli = SKChatClient.from_env(**kw.pop("client_kw", {}))
    if "temperature" in kw:
        cli.temperature = kw.pop("temperature")
    return await cli.complete(**kw)


def hint(e):
    s = e.status
    if s == 401 or s == 403:
        return "revisa AZURE_OPENAI_API_KEY (clave del mismo recurso que el endpoint) y que no tenga comillas ni espacios"
    if s == 404:
        return "revisa AZURE_OPENAI_DEPLOYMENT_NAME (nombre EXACTO del despliegue) y que el endpoint termine en /openai/v1/"
    if s == 400 and "temperature" in e.message.lower():
        return "este modelo no acepta temperature; deja LLM_TEMPERATURE vacio"
    if s == 429:
        return "limite de cuota; espera un minuto o sube la cuota del despliegue"
    return "revisa el mensaje del proveedor de arriba"


def main():
    if not load_env() and not os.environ.get("AZURE_OPENAI_ENDPOINT"):
        print("No encuentro .env en la raiz del proyecto."); return 1
    e = os.environ.get
    host = urlparse(e("AZURE_OPENAI_ENDPOINT", "")).netloc or "(vacio)"
    line(None, f"endpoint host: {host}   despliegue: {(e('AZURE_OPENAI_DEPLOYMENT_NAME') or e('AZURE_OPENAI_DEPLOYMENT')) or '(vacio)'}   clave: "
               f"{'configurada' if e('AZURE_OPENAI_API_KEY') and not e('AZURE_OPENAI_API_KEY').startswith(('REEMPLAZAR', 'pega_aqui', '<')) else 'FALTA o es marcador'}")
    key = e("PRICING_MODEL_KEY", "")
    rates = load_rates(key)
    if rates:
        line(True, f"tarifa {key}: entrada {rates['input']} / cacheada {rates['cached_input']} / salida {rates['output']} USD por 1M"
                   + ("  (PROVISIONAL: verifica en el calculador de precios de Azure)" if rates.get("provisional") else ""))
    else:
        line(False, f"PRICING_MODEL_KEY='{key}' no existe en config/pricing.json: no se podra calcular el costo")
    total, ok_all = 0.0, True

    def spend(r):
        nonlocal total
        c = cost_usd(rates, r.usage.prompt_tokens, r.usage.completion_tokens, r.usage.cached_tokens)
        total += c or 0

    # 1) conexion basica
    try:
        r = asyncio.run(probe("basico", messages=[{"role": "user", "content": "Responde solo con la palabra OK."}]))
        spend(r)
        line(True, f"conexion y despliegue: respuesta '{r.text[:40]}' en {r.latency_s:.1f}s; tokens entrada {r.usage.prompt_tokens}, "
                   f"salida {r.usage.completion_tokens} (razonamiento {r.usage.reasoning_tokens})")
        if r.usage.prompt_tokens == 0:
            line(False, "la respuesta no trae 'usage': sin eso no se pueden medir tokens ni costo")
            ok_all = False
    except LLMError as ex:
        line(False, f"conexion: {ex.message}\n         pista: {hint(ex)}")
        return 2
    # 2) tool calling
    try:
        r = asyncio.run(probe("tools", messages=[{"role": "system", "content": "Usa las tools cuando haga falta."},
                                                 {"role": "user", "content": "Inspecciona la imagen img_demo_01."}], tools=TOOL))
        spend(r)
        c = r.tool_calls[0] if r.tool_calls else None
        good = bool(c) and c.name == "inspect_image" and c.arguments.get("image_ids") == ["img_demo_01"]
        line(good, "llamada a tool: " + (f"{c.name}({json.dumps(c.arguments)})" if c else f"el modelo no llamo a la tool (texto: {r.text[:60]!r})"))
        ok_all &= good
    except LLMError as ex:
        line(False, f"llamada a tool: {ex.message}\n         pista: {hint(ex)}")
        ok_all = False
    # 3) temperatura 0
    try:
        r = asyncio.run(probe("temp", messages=[{"role": "user", "content": "Responde solo con la palabra OK."}], temperature=0.0))
        spend(r)
        line(True, "temperature=0 aceptada: puedes fijar LLM_TEMPERATURE=0 en .env para repetibilidad")
    except LLMError as ex:
        line(None, "temperature=0 NO aceptada por este despliegue; deja LLM_TEMPERATURE vacio. "
                   f"Detalle: {ex.message[:160]}")
    line(None, f"costo estimado de esta prueba: ${total:.6f} USD" if rates else "costo no calculable sin tarifa")
    print("\nListo: la conexion funciona." if ok_all else "\nHay fallas: corrigelas y vuelve a correr este script.")
    return 0 if ok_all else 3


if __name__ == "__main__":
    sys.exit(main())
