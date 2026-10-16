# Catálogo de tools y contratos

Generado por `python scripts/export_contracts.py` desde el código (versión de contrato de las tools: `1.0`). No editar a mano.

Todas las tools se exponen por **MCP** (JSON-RPC; stdio o streamable-HTTP) y comparten el mismo pipeline: identidad (JWT) → rol (RBAC) → línea → esquema JSON → idempotencia (escrituras) → tool → auditoría encadenada. Las respuestas son siempre datos: `{"ok": true, ...}` o `{"ok": false, "error": {code, message, retryable, http_status}}`; nunca excepciones.

La **identidad no es un argumento**: `user`, `role` y `line` salen del JWT. Si una tool recibe `line_id` y no coincide con la línea del JWT, responde `FORBIDDEN_LINE`.

## Resumen

| Tool | Servidor | Roles | Efecto |
|---|---|---|---|
| `inspect_image` | vision_server | todos | solo lectura |
| `queue_reject` | vision_server | supervisor, calidad | escritura (idempotente por sesion) |
| `list_review_queue` | vision_server | supervisor, calidad | solo lectura |
| `confirm_review_label` | vision_server | supervisor, calidad | escritura (idempotente por sesion) |
| `query_metrics_turn` | ops_server | todos | solo lectura |
| `get_line_context` | ops_server | todos | solo lectura |
| `generate_shift_report` | ops_server | supervisor | escritura (idempotente por sesion) |
| `change_confidence_threshold` | ops_server | supervisor | riesgo: solo prepara una solicitud; la decide una persona |
| `control_line_state` | ops_server | supervisor | riesgo: solo prepara una solicitud; la decide una persona |
| `send_manual_angle` | ops_server | supervisor | riesgo: solo prepara una solicitud; la decide una persona |
| `search_sop` | maintenance_server | mantenimiento, supervisor, ingeniero_ml | solo lectura |
| `create_ticket` | maintenance_server | supervisor, mantenimiento | escritura (idempotente por sesion) |
| `upload_new_sku_views` | ml_eng_server | ingeniero_ml | escritura (idempotente por sesion) |
| `test_sku_model` | ml_eng_server | ingeniero_ml | solo lectura |
| `promote_model_version` | ml_eng_server | ingeniero_ml | riesgo: solo prepara una solicitud; la decide una persona |

## Contratos de entrada

### `inspect_image`

Valida la calidad de 1 a 5 imagenes ya cargadas, identifica el objeto, estima el angulo y devuelve la decision GRASP / RETRY_VIEW / REJECT con confianza calibrada, incertidumbre y vistas similares como evidencia. Si la calidad no alcanza, no se llama al modelo y se pide volver a capturar. Solo lectura.

- Servidor: `vision_server` · Roles: todos · Efecto: solo lectura

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": [
    "image_ids"
  ],
  "properties": {
    "image_ids": {
      "type": "array",
      "minItems": 1,
      "maxItems": 5,
      "uniqueItems": true,
      "items": {
        "type": "string",
        "pattern": "^[A-Za-z0-9_-]{1,64}$"
      }
    },
    "line_id": {
      "type": "string"
    }
  }
}
```

### `queue_reject`

Deriva a la cola de revision humana una imagen ya inspeccionada (normalmente con decision REJECT) y devuelve su queue_id.

- Servidor: `vision_server` · Roles: supervisor, calidad · Efecto: escritura (idempotente por sesion)

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": [
    "image_id",
    "reason"
  ],
  "properties": {
    "image_id": {
      "type": "string",
      "pattern": "^[A-Za-z0-9_-]{1,64}$"
    },
    "reason": {
      "type": "string",
      "minLength": 3,
      "maxLength": 200
    },
    "line_id": {
      "type": "string"
    }
  }
}
```

### `list_review_queue`

Lista hasta 20 casos de la cola de revision humana de su linea.

- Servidor: `vision_server` · Roles: supervisor, calidad · Efecto: solo lectura

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "status": {
      "type": "string",
      "enum": [
        "open",
        "labeled",
        "manual_angle_sent"
      ]
    },
    "limit": {
      "type": "integer",
      "minimum": 1,
      "maximum": 20
    },
    "line_id": {
      "type": "string"
    }
  }
}
```

### `confirm_review_label`

Registra la etiqueta (clase y angulo opcional) que confirmo una persona para un caso de la cola y la deja en staging de reentrenamiento.

- Servidor: `vision_server` · Roles: supervisor, calidad · Efecto: escritura (idempotente por sesion)

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": [
    "queue_id",
    "confirmed_class"
  ],
  "properties": {
    "queue_id": {
      "type": "string",
      "pattern": "^Q-[a-z0-9]{1,16}$"
    },
    "confirmed_class": {
      "type": "integer",
      "minimum": 1,
      "maximum": 100000
    },
    "confirmed_angle": {
      "type": "number",
      "minimum": 0,
      "exclusiveMaximum": 360
    },
    "line_id": {
      "type": "string"
    }
  }
}
```

### `query_metrics_turn`

Consulta una metrica de un turno (throughput, rejects, grasp_failures, downtime), total o por hora, comparada con el turno anterior. Solo lectura.

- Servidor: `ops_server` · Roles: todos · Efecto: solo lectura

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": [
    "turn_id",
    "metric",
    "interval"
  ],
  "properties": {
    "turn_id": {
      "type": "string",
      "pattern": "^(\\d{4}-\\d{2}-\\d{2}-)?T[123]$",
      "description": "AAAA-MM-DD-T1..T3, o T1..T3 para el dia de hoy"
    },
    "metric": {
      "type": "string",
      "enum": [
        "throughput",
        "rejects",
        "grasp_failures",
        "downtime"
      ]
    },
    "interval": {
      "type": "string",
      "enum": [
        "turn",
        "hour"
      ]
    },
    "line_id": {
      "type": "string"
    }
  }
}
```

### `get_line_context`

Estado de la camara (calibracion, nitidez, exposicion, fallos), version de modelo y umbral vigentes, y cambios de version, umbral o estado dentro de una ventana de maximo 24 h. Solo lectura.

- Servidor: `ops_server` · Roles: todos · Efecto: solo lectura

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": [
    "window"
  ],
  "properties": {
    "window": {
      "oneOf": [
        {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "kind",
            "turn_id"
          ],
          "properties": {
            "kind": {
              "const": "turn"
            },
            "turn_id": {
              "type": "string",
              "pattern": "^(\\d{4}-\\d{2}-\\d{2}-)?T[123]$"
            }
          }
        },
        {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "kind",
            "start",
            "end"
          ],
          "properties": {
            "kind": {
              "const": "range"
            },
            "start": {
              "type": "string",
              "pattern": "^\\d{4}-\\d{2}-\\d{2}[T ]\\d{2}:\\d{2}(:\\d{2})?$"
            },
            "end": {
              "type": "string",
              "pattern": "^\\d{4}-\\d{2}-\\d{2}[T ]\\d{2}:\\d{2}(:\\d{2})?$"
            }
          }
        }
      ]
    },
    "line_id": {
      "type": "string"
    }
  }
}
```

### `generate_shift_report`

Genera (o reemplaza) el reporte de entrega de un turno ya cerrado: metricas, incidentes, tickets abiertos y casos pendientes.

- Servidor: `ops_server` · Roles: supervisor · Efecto: escritura (idempotente por sesion)

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": [
    "turn_id"
  ],
  "properties": {
    "turn_id": {
      "type": "string",
      "pattern": "^(\\d{4}-\\d{2}-\\d{2}-)?T[123]$"
    },
    "line_id": {
      "type": "string"
    }
  }
}
```

### `change_confidence_threshold`

Prepara el cambio del umbral de confianza de la linea. NO lo aplica: queda pendiente de aprobacion humana (devuelve request_id).

- Servidor: `ops_server` · Roles: supervisor · Efecto: riesgo: solo prepara una solicitud; la decide una persona

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": [
    "new_threshold",
    "reason"
  ],
  "properties": {
    "new_threshold": {
      "type": "number",
      "exclusiveMinimum": 0,
      "exclusiveMaximum": 1
    },
    "reason": {
      "type": "string",
      "minLength": 5,
      "maxLength": 300
    },
    "line_id": {
      "type": "string"
    }
  }
}
```

### `control_line_state`

Prepara pausar o reanudar la linea. NO lo aplica: queda pendiente de aprobacion humana (devuelve request_id).

- Servidor: `ops_server` · Roles: supervisor · Efecto: riesgo: solo prepara una solicitud; la decide una persona

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": [
    "action",
    "reason"
  ],
  "properties": {
    "action": {
      "type": "string",
      "enum": [
        "pause",
        "resume"
      ]
    },
    "reason": {
      "type": "string",
      "minLength": 5,
      "maxLength": 300
    },
    "line_id": {
      "type": "string"
    }
  }
}
```

### `send_manual_angle`

Prepara el envio de un angulo manual al robot para un caso de la cola de revision. NO lo envia: requiere aprobacion humana (devuelve request_id).

- Servidor: `ops_server` · Roles: supervisor · Efecto: riesgo: solo prepara una solicitud; la decide una persona

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": [
    "image_id",
    "manual_angle",
    "reason"
  ],
  "properties": {
    "image_id": {
      "type": "string",
      "pattern": "^[A-Za-z0-9_-]{1,64}$"
    },
    "manual_angle": {
      "type": "number",
      "minimum": 0,
      "exclusiveMaximum": 360
    },
    "reason": {
      "type": "string",
      "minLength": 5,
      "maxLength": 300
    },
    "line_id": {
      "type": "string"
    }
  }
}
```

### `search_sop`

Busca en los SOP y manuales y devuelve hasta 3 fragmentos truncados con sus document_ids para citarlos. Si no hay resultados, no hay SOP aplicable. Solo lectura.

- Servidor: `maintenance_server` · Roles: mantenimiento, supervisor, ingeniero_ml · Efecto: solo lectura

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": [
    "query"
  ],
  "properties": {
    "query": {
      "type": "string",
      "minLength": 3,
      "maxLength": 300
    },
    "equipment": {
      "type": "string",
      "enum": [
        "gripper",
        "camera",
        "robot_arm",
        "conveyor",
        "lighting"
      ]
    }
  }
}
```

### `create_ticket`

Crea un ticket de mantenimiento (baja, media o alta). Si ya hay uno abierto igual para la linea y el equipo dentro de la ventana, devuelve el existente.

- Servidor: `maintenance_server` · Roles: supervisor, mantenimiento · Efecto: escritura (idempotente por sesion)

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": [
    "issue_desc",
    "priority",
    "equipment"
  ],
  "properties": {
    "issue_desc": {
      "type": "string",
      "minLength": 10,
      "maxLength": 500
    },
    "priority": {
      "type": "string",
      "enum": [
        "baja",
        "media",
        "alta"
      ]
    },
    "equipment": {
      "type": "string",
      "enum": [
        "gripper",
        "camera",
        "robot_arm",
        "conveyor",
        "lighting"
      ]
    },
    "line_id": {
      "type": "string"
    }
  }
}
```

### `upload_new_sku_views`

Valida un lote de vistas de un SKU nuevo (cantidad minima, calidad de cada imagen y cobertura de angulos) y lo deja en staging. Nunca toca el modelo productivo.

- Servidor: `ml_eng_server` · Roles: ingeniero_ml · Efecto: escritura (idempotente por sesion)

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": [
    "sku_name",
    "image_ids"
  ],
  "properties": {
    "sku_name": {
      "type": "string",
      "pattern": "^[A-Za-z0-9][A-Za-z0-9_-]{1,39}$"
    },
    "image_ids": {
      "type": "array",
      "minItems": 1,
      "maxItems": 200,
      "uniqueItems": true,
      "items": {
        "type": "string",
        "pattern": "^[A-Za-z0-9_-]{1,64}$"
      }
    },
    "line_id": {
      "type": "string"
    }
  }
}
```

### `test_sku_model`

Evalua en staging el modelo candidato con el SKU nuevo segun el protocolo A, B, C o D y devuelve exactitud, error angular circular medio, % dentro de 10 grados y un evidence_report_id.

- Servidor: `ml_eng_server` · Roles: ingeniero_ml · Efecto: solo lectura

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": [
    "sku_name",
    "protocol"
  ],
  "properties": {
    "sku_name": {
      "type": "string",
      "pattern": "^[A-Za-z0-9][A-Za-z0-9_-]{1,39}$"
    },
    "protocol": {
      "type": "string",
      "enum": [
        "A",
        "B",
        "C",
        "D"
      ]
    },
    "line_id": {
      "type": "string"
    }
  }
}
```

### `promote_model_version`

Prepara la promocion de una version candidata a produccion. NO la aplica: el middleware valida la evidencia contra los minimos y la solicitud queda pendiente de aprobacion de otro usuario.

- Servidor: `ml_eng_server` · Roles: ingeniero_ml · Efecto: riesgo: solo prepara una solicitud; la decide una persona

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": [
    "candidate_version",
    "evidence_report_id",
    "reason"
  ],
  "properties": {
    "candidate_version": {
      "type": "string",
      "minLength": 3,
      "maxLength": 80
    },
    "evidence_report_id": {
      "type": "string",
      "pattern": "^ER-[a-f0-9]{8}$"
    },
    "reason": {
      "type": "string",
      "minLength": 5,
      "maxLength": 300
    },
    "line_id": {
      "type": "string"
    }
  }
}
```

## Códigos de error

| Código | HTTP | Reintentable | Significado |
|---|---|---|---|
| `UNAUTHENTICATED` | 401 | no | JWT ausente, invalido o vencido |
| `FORBIDDEN` | 403 | no | el rol del JWT no puede usar la tool |
| `FORBIDDEN_LINE` | 403 | no | el argumento line_id no coincide con la linea del JWT |
| `UNKNOWN_TOOL` | 404 | no | la tool no existe en ese servidor (o no esta en el conjunto del agente) |
| `VALIDATION_ERROR` | 422 | no | los argumentos no cumplen el esquema JSON o una regla de negocio; incluye el campo |
| `IMAGE_NOT_FOUND` | 404 | no | alguna imagen no existe en la linea del usuario (incluye missing_ids) |
| `VISION_TIMEOUT` | 504 | no | el servicio de vision no respondio dentro del limite (no se reintenta automaticamente) |
| `VISION_UNREACHABLE` | 502 | sí | no se pudo conectar con el servicio de vision |
| `VISION_5XX` | 502 | sí | el servicio de vision respondio 5xx (la tool ya hizo su reintento controlado) |
| `VISION_REJECTED_REQUEST` | 422 | no | el servicio de vision rechazo la solicitud (4xx) |
| `NOT_INSPECTED` | 409 | no | queue_reject solo acepta imagenes ya inspeccionadas |
| `NOT_FOUND` | 404 | no | caso inexistente en la linea |
| `UNKNOWN_CLASS` | 422 | no | la clase no esta en el catalogo de SKU |
| `CONFLICT` | 409 | no | el caso ya tiene otra etiqueta confirmada; no se sobrescribe |
| `CASE_NOT_OPEN` | 409 | no | el caso no esta abierto |
| `TURN_NOT_FOUND` | 404 | no | el turno no existe para la linea |
| `TURN_NOT_CLOSED` | 409 | no | el reporte solo se genera con el turno cerrado |
| `NOT_IN_REVIEW_QUEUE` | 409 | no | el caso debe estar en la cola de revision de la linea |
| `CASE_NOT_ACTIONABLE` | 409 | no | el caso no esta en un estado que permita la accion |
| `SKU_EXISTS` | 409 | no | ya existe un SKU con ese nombre |
| `NOT_VALIDATED` | 409 | no | el SKU no tiene un lote validado en staging |
| `NO_EVIDENCE_AVAILABLE` | 501 | no | el MVP solo evalua SKU de demostracion (simulado) |
| `EVIDENCE_NOT_FOUND` | 404 | no | la evidencia no existe o no corresponde a la version candidata |
| `NOT_A_CANDIDATE` | 409 | no | la version no esta como candidata |
| `EVIDENCE_BELOW_MINIMUMS` | 422 | no | la evidencia no alcanza los minimos de promocion; no se crea la solicitud |
| `INTERNAL_ERROR` | 500 | no | fallo inesperado; el detalle queda solo en la auditoria, no se devuelve al LLM |

Reintentable = el agente puede repetir la misma llamada una vez; el orquestador además corta un segundo intento idéntico (`loop_detected`) y el máximo de 4 rondas de tools (`max_iterations`).

## Contratos de las APIs HTTP

- Servicio de visión: `docs/contracts/vision_api.openapi.json` (`POST /v1/predict`, `GET /model`, `GET /health`).
- API del agente y aprobaciones: `docs/contracts/agent_api.openapi.json` (`POST /v1/chat`, `/v1/images`, `/v1/approvals/...`).
