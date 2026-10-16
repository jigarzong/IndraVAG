# Prueba Técnica: Líder Técnico de IA (Indra)

Solución técnica de extremo a extremo para reemplazar tareas manuales en una línea de producción semiautomática con Visión Artificial, Machine Learning clásico y arquitecturas agénticas.

## Entregables

| Reto | Entregable | Ubicación |
| :--- | :--- | :--- |
| Reto 1 | Diagramas AS-IS y TO-BE (BPMN) | [`Reto 1 - Modelado AS-IS y TO-BE.pdf`](<./Reto 1 - Modelado AS-IS y TO-BE.pdf>) (raíz del repositorio) |
| Reto 2 | Notebook reproducible (CRISP-DM, secciones 9 en adelante vigentes) | [`notebooks/Indra_MLModel_Vision.ipynb`](./notebooks/Indra_MLModel_Vision.ipynb) |
| Reto 2 | Reporte de resultados por protocolo, análisis de errores y model card (PDF) | [`reports/Reto 2 - Resultados, analisis de errores y model card.pdf`](./reports) |
| Reto 2 | Model Card | [`model_card.md`](./model_card.md) |
| Reto 2 | Modelo exportado y función de inferencia | [`models/protocolo_A/`](./models/protocolo_A), [`src/inference.py`](./src/inference.py) |
| Reto 3 | Arquitectura agéntica, herramientas y diagramas | [`Reto 3 - Asistente Agentico de Operacion de Vision.pdf`](<./Reto 3 - Asistente Agentico de Operacion de Vision.pdf>) y [`.docx`](<./Reto 3 - Asistente Agentico de Operacion de Vision.docx>) (raíz); la implementación está en [`docs/`](./docs) y [`services/`](./services). Las secciones 7.6 (resultados medidos) y 7.7 del documento están pendientes de actualizar con la evaluación completa |
| Reto 4 | MVP ejecutable: servicio de visión (API propia), 4 servidores MCP, API del agente con pantalla y aprobaciones | [`services/`](./services), [`docker-compose.yml`](./docker-compose.yml), [`scripts/run_stack.py`](./scripts/run_stack.py) |
| Reto 4 | Arquitectura, diagramas de componentes y de secuencia (imagen→robot; flujo agéntico con aprobación humana) | [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md), [`docs/diagrams/`](./docs/diagrams) |
| Reto 4 | Catálogo de tools y contratos (Markdown, JSON Schema, OpenAPI) | [`docs/TOOLS_CATALOG.md`](./docs/TOOLS_CATALOG.md), [`docs/contracts/`](./docs/contracts) |
| Reto 4 | Modelo de amenazas y controles | [`docs/THREAT_MODEL.md`](./docs/THREAT_MODEL.md) |
| Reto 4 | Demo de los 8 escenarios adversos con traza | `python scripts/demo_scenarios.py` (escribe `demo/`) |
| Reto 5 | Plan de proyecto, fases y RACI | Pendiente |
| Reto 6 | Arquitectura Cloud (AWS vs Azure) y FinOps | Pendiente |

## Estructura
```
src/        lib.py (descriptores, protocolos, ángulos), experiment.py (validación, umbrales, run_protocol),
            run_protocols.py, baselines.py, analysis.py, prepare_data.py, inference.py, export_model.py, log_mlflow.py
results/    JSON por protocolo, salidas de prueba, figuras y análisis de errores
models/     protocolo_A/bundle.joblib + thresholds.json
notebooks/  Indra_MLModel_Vision.ipynb
data/       COIL-100 (no versionado) y data/cache/ (generado)
```

## Reproducción (Reto 2)
1. `pip install numpy pandas matplotlib opencv-python scikit-learn scikit-image scipy joblib mlflow tensorflow-cpu nbformat`
2. Colocar COIL-100 en `data/COIL-100` y ejecutar `INDRA_DATA=data/COIL-100 python src/prepare_data.py` (debe imprimir la huella `26635cc67c13f12b4010941d0e9e9af0bfda30c52007040377be0caba01c4c4a`).
3. `FS=hogcolor python src/run_protocols.py A B C D` y `FS=pix python src/run_protocols.py A B C D` (≈1 h en 2 núcleos). Referencias: `python src/baselines.py nn A B C D` y `python src/baselines.py cnn A B C D`.
4. `python src/analysis.py` (figuras y análisis de errores) y `python src/log_mlflow.py` (registro en MLflow, backend SQLite).
5. Notebook: ejecutar desde `notebooks/`; con `RERUN = False` carga los resultados guardados.

Semilla 42 en todos los pasos. Las secciones 4–8 del notebook son el primer intento (conservadas como historial) y sus cifras están reemplazadas por la sección 9.

## Servicios (Retos 3 y 4) — avance

```
services/vision_api/      API de visión independiente del agente: POST /v1/predict, GET /health, GET /model
services/common/          middleware compartido: identidad JWT, RBAC, esquemas JSON, idempotencia, auditoría encadenada, aprobaciones (HITL), datos simulados
services/mcp_servers/     4 servidores MCP con las 15 tools: vision, ops, maintenance, ml_eng
services/agent/           orquestador: router + 4 agentes ReAct (Semantic Kernel), cliente LLM, acceso a las tools por MCP, memoria de 5 turnos, costo por solicitud
eval/                     30 casos reproducibles (cases.jsonl), criterios deterministas (checks.py) y el comando que los ejecuta (run_eval.py)
services/agent_api/       API del agente (login demo, carga de imágenes, chat, aprobaciones, auditoría) y la pantalla web (static/index.html)
docs/                     ARCHITECTURE.md, THREAT_MODEL.md, TOOLS_CATALOG.md, contracts/ (generados del código), diagrams/ (Mermaid + PNG)
scripts/                  run_stack.py (todo sin Docker), bootstrap.py, seed_demo.py, init_env.py, check_llm.py, smoke_stack.py, demo_scenarios.py, export_contracts.py
config/pricing.json       tarifas del LLM (USD por 1M de tokens); las de gpt-5.4-mini son PROVISIONALES (fuente de terceros)
tests/                    pruebas (pytest): `python -m pytest tests -q`
deploy/ + docker-compose.yml   imágenes y compose (NO probados: sin demonio Docker en la sesión de desarrollo; validar con `docker compose config` y `docker compose up --build`)
.env.example              toda la configuración; sin secretos
```

Ejecución local:
```
pip install -r requirements-dev.txt        # incluye Semantic Kernel
python scripts/init_env.py                 # crea .env con JWT_SECRET y APPROVAL_SECRET aleatorios (si no existe)
python src/prepare_data.py                 # cache de COIL-100 (ver "Reproducción (Reto 2)")
python scripts/check_llm.py                # prueba de conexión al LLM (no imprime la clave)
python scripts/seed_demo.py --reset        # base simulada + imágenes de demostración + cola de revisión
python -m pytest tests -q
```

Levantar el MVP completo (visión + 4 servidores MCP + API del agente y pantalla):
```
python scripts/run_stack.py                # sin Docker; pantalla en http://127.0.0.1:8000/  (Ctrl+C detiene todo)
python scripts/smoke_stack.py              # prueba de humo (agrega --chat para una consulta real al LLM)
docker compose up --build                  # con Docker (misma pantalla; requiere .env)
python scripts/demo_scenarios.py           # los 8 escenarios adversos con traza (LLM real; usa tokens)
```
Usuarios demo de la pantalla: `ana.supervisor`, `luis.supervisor` (aprobar lo que pide el otro), `carla.calidad`, `mario.mantenimiento`, `vega.ingeniero_ml`, `pablo.ingeniero_ml`, `rosa.supervisor_l2` (otra línea). `AUTH_MODE=demo` solo es para la demostración.

Evaluación (Reto 3, sección 7.6):
```
python eval/run_eval.py                    # 30 casos, agente múltiple, LLM real -> eval/results.csv, .md y .json
python eval/run_eval.py --mode both        # además el agente único con las 15 tools (comparación de arquitectura)
python eval/run_eval.py --only insp-01,risk-02
python eval/run_eval.py --llm oracle       # AUTOPRUEBA del arnés con la solución de referencia: no son resultados
```
Cada caso corre con su propia copia de la base y de las imágenes sembradas, los 4 servidores MCP reales por stdio y el servicio de visión real (con instancias de falla 5xx y timeout para los casos del escenario 4). Mide éxito, selección de tools (orden y argumentos), seguridad (ningún cambio de línea, robot o modelo sin aprobación), tokens, costo y latencia. El éxito se decide con criterios deterministas (palabras esperadas, valores devueltos por las tools, estado final de la base), sin un LLM como juez.

Decisiones y límites de este tramo
* `inspect_image` valida la calidad antes de llamar al modelo (si falla, devuelve `RECAPTURE` y no hay predicción), acepta 1 a 5 imágenes y oculta la clase y el ángulo cuando la decisión es `REJECT`.
* La confianza calibrada del modelo sale ≈ 1.0 incluso en imágenes que el servicio rechaza (el rechazo lo causan el puntaje crudo de clase por debajo de 0.503 y la distancia a las vistas conocidas). Por eso las tools devuelven también `class_score`, `distance_to_known_views` y un motivo (`LOW_CLASS_SCORE`, `FAR_FROM_KNOWN_VIEWS`, `ANGLE_UNCERTAIN`, `BELOW_LINE_THRESHOLD`).
* Con los umbrales del protocolo A la cobertura en prueba es 45,2 %: más de la mitad de las imágenes de COIL-100 se rechazan o piden otra vista.
* Las operaciones de riesgo solo crean una solicitud (`pending_human_approval`). La aprobación la hace una persona con otra credencial (`services/common/approvals.py`); el resultado de las tools nunca contiene el token.  La API y la pantalla de aprobación están en `services/agent_api/`.
* Límites del orquestador aplicados por código: máximo 4 iteraciones (`MAX_AGENT_ITERATIONS`), corte ante dos llamadas idénticas, cada agente ve solo las tools de su rol, identidad y línea salen del JWT. El prompt no es la defensa.
* El LLM se llama por la API v1 de Azure AI Foundry (`AZURE_OPENAI_ENDPOINT` termina en `/openai/v1/`; el nombre del despliegue va en `AZURE_OPENAI_DEPLOYMENT_NAME`). `LLM_TEMPERATURE=0` solo si el despliegue lo acepta (`check_llm.py` lo comprueba).
* Simulado: métricas, estado de cámara, SOPs (uno de ellos lleva una instrucción oculta a propósito, escenario 8), tickets, robot y `test_sku_model` (usa como referencia las métricas medidas del Reto 2; no evalúa un SKU real). El MVP no usa el reloj para los turnos (`DEMO_TODAY`) para que los casos sean reproducibles.
* La búsqueda de SOPs usa TF-IDF léxico, no embeddings; la promoción de versión solo actualiza el registro (el servicio de visión sigue sirviendo el artefacto exportado).
* `mcp>=1.26,<2.0`: Semantic Kernel 1.45 exige `mcp<2`.
