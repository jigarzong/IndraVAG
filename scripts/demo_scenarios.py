"""Demo de los 8 escenarios adversos con traza legible (LLM real por defecto):

  python scripts/demo_scenarios.py                 # LLM real (usa tokens); escribe demo/demo_scenarios.{csv,md,json}
  python scripts/demo_scenarios.py --llm oracle    # sin LLM: solo comprueba que el montaje de la demo funciona (no es una demostracion)
  python scripts/demo_scenarios.py --only risk-02  # un caso

Es el mismo arnes de eval/run_eval.py restringido a los casos con escenario (1 fuera de dominio, 2 confianza baja, 3 argumentos invalidos,
4 falla/timeout del servicio de vision, 5 loop agentico, 6 operacion de riesgo sin aprobacion, 7 otra linea, 8 instruccion oculta en un SOP).
Cada caso corre sobre su propia copia de los datos, con los servidores MCP reales y el servicio de vision real (con fallas inyectadas donde aplica).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import run_eval                                                  # noqa: E402

if __name__ == "__main__":
    extra = ["--scenarios", "--trace"]
    if not any(x.startswith("--out") for x in sys.argv[1:]):
        extra += ["--out", "demo/demo_scenarios.csv"]
    sys.argv = [sys.argv[0], *extra, *sys.argv[1:]]
    run_eval.main()
