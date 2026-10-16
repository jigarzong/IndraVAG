# Modelo de amenazas y controles (entregable 8)

Alcance: el asistente agéntico del MVP (API del agente, orquestador, servidores MCP, servicio de visión, datos). Método: se enumeran activos y fronteras de confianza, y para cada amenaza se indica el control **implementado**, dónde está y qué prueba lo cubre. Lo que no está cubierto se declara como riesgo residual. No es una auditoría de seguridad ni un análisis de penetración.

## 1. Activos y fronteras de confianza

Activos: estado de la línea (umbral, pausa, versión de modelo), cola de revisión y robot, imágenes, registro de auditoría, secretos (`JWT_SECRET`, `APPROVAL_SECRET`, clave del LLM).

Fronteras: (1) navegador → API del agente; (2) agente → LLM (servicio externo); (3) agente → servidores MCP; (4) MCP → servicio de visión y datos; (5) **contenido recuperado** (SOP, texto de usuario, resultados de tools) → contexto del LLM: este contenido **no es de confianza**.

Principio rector: el LLM es un componente no confiable que *propone*; el código decide qué se permite.

## 2. Amenazas y controles

| # | Amenaza | Control implementado | Dónde | Prueba |
|---|---|---|---|---|
| T1 | Suplantación: llamar a la API o a una tool sin identidad | JWT firmado (HS256) obligatorio; secretos de ≥ 32 caracteres (la API no arranca sin ellos) | `common/auth.py`, `agent_api/app.py` | `test_middleware.py::test_missing_or_forged_token`, `::test_expired_token`, `test_agent_api.py::test_requires_token_and_unknown_user` |
| T2 | El LLM o el usuario escogen otra identidad/línea (p. ej. "consulta la línea 2") | Identidad solo del JWT; `line_id` distinto → `FORBIDDEN_LINE` | `common/toolkit.py` | caso `ops-03` (escenario 7), `test_middleware.py::test_other_line_is_denied_and_isolated`, `test_agent.py::test_other_line_is_denied_by_middleware` |
| T3 | Escalada de privilegios por rol (calidad intenta pausar la línea) | RBAC por tool + el agente solo recibe las tools de su especialidad (doble barrera) | `toolkit.py`, `agent/prompts.py` | caso `risk-04`, `test_middleware.py::test_rbac_blocks_wrong_role`, `test_agent.py::test_agent_cannot_call_tools_outside_its_role` |
| T4 | Acción de riesgo sin aprobación ("ya lo aprobé con el jefe") | Las tools de riesgo solo crean una solicitud; la aprobación es un endpoint HTTP distinto, autenticado, con token HMAC; no es tool MCP, el LLM no ve el token | `common/approvals.py`, `effects.py` | `test_approvals.py`, casos `risk-01..05`, `test_agent_api.py::test_risk_flow…` |
| T5 | Aprobación indebida: otro rol, otra línea, caducada, repetida, contexto cambiado | Rol, línea, TTL (30 min), estado `pending`, hash del contexto; promoción exige aprobador ≠ solicitante; sin pistas sobre solicitudes de otras líneas (404) | `approvals.py` | `test_approvals.py`, `test_agent_api.py` |
| T6 | Inyección de instrucciones indirecta (un SOP que ordena "pausa la línea") | Regla en los prompts (el texto recuperado es dato) y la tool `search_sop` marca los textos con instrucciones incrustadas; las acciones de riesgo no se ejecutan sin aprobación humana aunque el LLM obedezca; evaluación del caso `sop-03` | `agent/prompts.py`, `maintenance_server.py`, `approvals.py` | `test_tools.py::test_search_sop_flags_embedded_instructions`, `test_agent.py::test_sop_injection_is_flagged_and_does_not_trigger_actions`; caso `sop-03` (**resultado con el LLM real pendiente de la corrida de evaluación**) |
| T7 | Inyección por imagen | La imagen nunca llega al LLM: solo identificadores y números devueltos por las tools | `agent_api/app.py`, `vision_server.py` | revisión de diseño |
| T8 | Archivo malicioso o enorme en la carga | Límite de 5 MB y 5 imágenes, decodificación con OpenCV, **re-codificación a PNG**, id generado (nunca se usa el nombre original) | `agent_api/app.py::upload` | `test_agent_api.py::test_upload_chat_with_image_and_validation` |
| T9 | Argumentos inválidos o maliciosos hacia las tools (inyección SQL, rutas, rangos) | Esquema JSON estricto (`additionalProperties: false`, patrones, rangos); consultas SQL parametrizadas; errores estructurados que el LLM puede corregir | `toolkit.py` | `test_middleware.py::test_schema_validation`, `test_tools.py::test_inspect_rejects_path_traversal_and_too_many`, caso `sku-03` |
| T10 | Bucles y consumo descontrolado (costo, disponibilidad) | 4 rondas de tools, corte de llamada idéntica repetida, timeout de 180 s, mensaje ≤ 2000 caracteres, tools con timeout y un reintento solo ante 5xx | `agent/orchestrator.py`, `vision_server.py` | `test_agent.py::test_identical_repeated_call_is_cut`, `::test_max_iterations_cut_with_failure_report`, `test_tools.py::test_vision_5xx_retries_once_then_structured_error`, casos `rob-03`, `insp-05/06` |
| T11 | Duplicados por reintentos (dos tickets, dos etiquetas) | Idempotencia por (tool, argumentos, sesión, línea) en escrituras no riesgosas; solicitudes de riesgo deduplicadas por hash | `toolkit.py`, `approvals.py` | `test_middleware.py::test_idempotent_write_replays_same_result` |
| T12 | Repudio / manipulación del registro | Auditoría con cadena de hashes, verificable por API; los argumentos largos se truncan a 200 caracteres | `common/audit.py` | `test_middleware.py::test_audit_chain_and_content`, `::test_audit_is_append_only_and_tamper_evident`, `test_agent_api.py` (verify) |
| T13 | Fuga de secretos | Secretos solo en `.env` (excluido por `.gitignore` y `.dockerignore`) o en secretos de la plataforma; `scripts/check_llm.py` solo informa si la clave está configurada, nunca su valor; el detalle de excepciones internas no se devuelve al LLM (`INTERNAL_ERROR`, solo queda en la auditoría) | `llm.py`, `toolkit.py`, `scripts/check_llm.py` | `test_agent.py::test_sk_client_rejects_placeholder_key_and_missing_deployment` (marcadores de clave); sin prueba específica del texto de `INTERNAL_ERROR` |
| T14 | Fallo del servicio de visión (5xx / timeout) | Errores estructurados con `retryable`; el agente informa y escala en lugar de inventar | `vision_server.py` | casos `insp-05`, `insp-06` |
| T15 | Salida del LLM con un resultado "inventado" | Los casos de evaluación exigen que los valores citados aparezcan en el resultado de la tool (`answer_results`); la prueba es determinista, sin LLM como juez | `eval/checks.py` | `test_eval.py` (**cumplimiento con el LLM real pendiente**) |

## 3. Riesgos residuales (no cubiertos o cubiertos parcialmente)

1. **`AUTH_MODE=demo`** permite pedir un JWT para cualquier usuario de ejemplo sin contraseña. Es solo para la demostración; en un despliegue real debe apagarse (cualquier otro valor lo desactiva) y usarse SSO.
2. **Auto-aprobación**: para cambiar el umbral, pausar la línea o enviar un ángulo, el solicitante con rol de supervisor *puede* aprobar su propia solicitud; solo la promoción de modelo exige un aprobador distinto. Recomendación: exigirlo también para las otras tres si la política de la planta lo pide (es un cambio de una línea en `REQUIRES_DISTINCT_APPROVER`).
3. **Sin TLS ni límite de tasa** en el MVP. Debe ir detrás de un API gateway con TLS, cuotas por usuario y protección contra abuso.
4. **Secreto compartido (HS256)** entre servicios: quien lo obtenga puede acuñar tokens. En producción: claves asimétricas del proveedor de identidad y rotación.
5. **SQLite compartido y sesiones en memoria**: sin alta disponibilidad ni aislamiento entre inquilinos.
6. **El LLM es externo**: los textos de la conversación y los resultados de las tools se envían al servicio de Azure. No se envían imágenes. Revisar la política de retención del despliegue.
7. **Inyección de instrucciones**: ningún control de prompts es infalible. La defensa real es de arquitectura (T4/T5): aunque el modelo obedezca un texto malicioso, no puede ejecutar acciones de riesgo; sí podría producir una respuesta engañosa o crear solicitudes pendientes innecesarias que una persona rechazaría.
8. **No se hizo prueba de penetración ni revisión externa.**
