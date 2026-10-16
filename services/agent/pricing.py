"""Costo estimado por llamada al LLM a partir de config/pricing.json (USD por 1M de tokens)."""
import json
from pathlib import Path

PRICING_FILE = Path(__file__).resolve().parents[2] / "config" / "pricing.json"


def load_rates(key):
    """Devuelve dict(input, cached_input, output, provisional?) o None si la clave no existe o no esta configurada."""
    if not key:
        return None
    try:
        models = json.loads(PRICING_FILE.read_text(encoding="utf-8"))["models"]
    except (OSError, ValueError, KeyError):
        return None
    return models.get(key)


def cost_usd(rates, prompt_tokens, completion_tokens, cached_tokens=0):
    """costo = (prompt - cacheados) * tarifa_entrada + cacheados * tarifa_cacheada + salida * tarifa_salida, por 1M de tokens.
    Los tokens de razonamiento ya vienen incluidos en completion_tokens, que es lo que factura el servicio."""
    if not rates:
        return None
    fresh = max(prompt_tokens - cached_tokens, 0)
    return (fresh * rates["input"] + cached_tokens * rates["cached_input"] + completion_tokens * rates["output"]) / 1_000_000
