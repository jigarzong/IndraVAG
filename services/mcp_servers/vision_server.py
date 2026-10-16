"""Servidor MCP de Vision (agente de Vision): inspect_image, queue_reject, list_review_queue, confirm_review_label.

El LLM nunca recibe bytes de imagen: las tools reciben identificadores que resuelven contra el almacen de imagenes y
solo se devuelve el resultado estructurado (sin pixeles, vectores ni probabilidades por clase).
"""
import hashlib
import json
import time
import uuid
from pathlib import Path

import httpx

from services.common import audit
from services.common.toolkit import ToolError, ToolServer, ToolSpec

ID = {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,64}$"}
ACTION = {"GRASP": "PROCEED", "RETRY_VIEW": "RETAKE_FROM_ANOTHER_ANGLE", "REJECT": "QUEUE_FOR_HUMAN_REVIEW"}


CLIENT_FACTORY = None          # las pruebas y el sembrado inyectan un cliente en proceso (TestClient) sin abrir sockets


def call_vision(cfg, files, client=None):
    """POST /v1/predict con timeout. Un reintento con espera solo ante 5xx o fallo de conexion (lectura idempotente);
    un timeout no se reintenta. Devuelve (json, None) o (None, ToolError)."""
    own = client is None
    client = client or httpx.Client(base_url=cfg.vision_url, timeout=cfg.vision_timeout_s)
    try:
        for attempt in range(cfg.vision_retries_5xx + 1):
            try:
                r = client.post("/v1/predict", files=files)
            except httpx.TimeoutException:
                return None, ToolError("VISION_TIMEOUT", f"el servicio de vision no respondio en {cfg.vision_timeout_s:g} s", False, 504)
            except httpx.TransportError as e:
                if attempt < cfg.vision_retries_5xx:
                    time.sleep(cfg.vision_retry_wait_s)
                    continue
                return None, ToolError("VISION_UNREACHABLE", f"no se pudo conectar al servicio de vision ({type(e).__name__})", True, 502)
            if r.status_code >= 500:
                if attempt < cfg.vision_retries_5xx:
                    time.sleep(cfg.vision_retry_wait_s)
                    continue
                return None, ToolError("VISION_5XX", f"el servicio de vision respondio {r.status_code}", True, 502, upstream_status=r.status_code)
            if r.status_code >= 400:
                return None, ToolError("VISION_REJECTED_REQUEST", f"solicitud rechazada ({r.status_code})", False, 422, upstream_status=r.status_code)
            return r.json(), None
    finally:
        if own:
            client.close()


def inspect_image(ctx, a):
    ids = a["image_ids"]
    rows = {}
    for i in ids:
        r = ctx.conn.execute("SELECT * FROM images WHERE image_id=? AND line_id=?", (i, ctx.ident.line_id)).fetchone()
        if r:
            rows[i] = r
    missing = [i for i in ids if i not in rows]
    if missing:                                                          # no se distingue "no existe" de "es de otra linea"
        raise ToolError("IMAGE_NOT_FOUND", "algunas imagenes no existen en su linea", False, 404, missing_ids=missing)
    paths = {i: ctx.cfg.image_store / rows[i]["path"] for i in ids}
    files = [("files", (paths[i].name, paths[i].read_bytes(), "application/octet-stream")) for i in ids]
    data, e = call_vision(ctx.cfg, files, client=CLIENT_FACTORY() if CLIENT_FACTORY else None)
    if e:
        e.extra["suggested_action"] = "queue_reject"                     # el agente informa y ofrece derivar; no inventa resultado
        raise e
    lc = ctx.conn.execute("SELECT * FROM line_config WHERE line_id=?", (ctx.ident.line_id,)).fetchone()
    line_thr = lc["confidence_threshold"] if lc else 0.0
    out, fingerprints = [], {}
    for iid, it in zip(ids, data["results"]):
        fingerprints[iid] = rows[iid]["sha256"][:16]
        if it["error"]:
            out.append(dict(image_id=iid, status="INVALID_IMAGE", message=it["error"]["message"], recommended_action="RECAPTURE"))
            continue
        if it.get("recommended_action") == "RECAPTURE":
            out.append(dict(image_id=iid, status="QUALITY_FAILED", quality=it["quality"], recommended_action="RECAPTURE",
                            note="no se llamo al modelo: la imagen no cumple los minimos de calidad"))
            continue
        p = it["prediction"]
        decision, reason = p["decision"], p["decision_reason"]
        if decision != "REJECT" and p["confidence"] < line_thr:          # el umbral de linea solo puede endurecer la decision del servicio
            decision, reason = "REJECT", "BELOW_LINE_THRESHOLD"
        res = dict(image_id=iid, status="OK", quality=it["quality"], decision=decision, decision_reason=reason,
                   recommended_action=ACTION[decision], confidence=round(p["confidence"], 3), class_score=round(p["class_score"], 3),
                   distance_to_known_views=round(max(p["nearest_view_distance"], 0.0), 3),
                   confidence_note="confidence = P(clase correcta) si el objeto esta en el catalogo; con objetos desconocidos no es fiable: mande la decision",
                   angle_uncertainty_deg=round(p["angle_uncertainty_deg"], 1))
        if decision == "REJECT":
            res.update(object_id=None, angle_deg=None, angle_reliable=False, similar_views=[])
        else:
            res.update(object_id=p["object_id"], angle_deg=round(p["angle_deg"], 1), angle_reliable=decision == "GRASP",
                       similar_views=[dict(object_id=s["object_id"], angle_deg=s["angle_deg"], similarity=round(s["cosine_similarity"], 3))
                                      for s in it["evidence"][:5]])
        out.append(res)
        ctx.conn.execute("INSERT OR REPLACE INTO inspections(image_id,line_id,result_json,model_version,created_at) VALUES(?,?,?,?,?)",
                         (iid, ctx.ident.line_id, json.dumps(res, ensure_ascii=False), p["thresholds_version"], audit.now_iso()))
    return dict(results=out, model_version=data["thresholds_version"], line_confidence_threshold=line_thr,
                _audit=dict(model_version=data["thresholds_version"], image_fingerprints=fingerprints,
                            decisions={r["image_id"]: r.get("decision", r["status"]) for r in out}))


def queue_reject(ctx, a):
    L = ctx.ident.line_id
    insp = ctx.conn.execute("SELECT * FROM inspections WHERE image_id=? AND line_id=?", (a["image_id"], L)).fetchone()
    if not insp:
        raise ToolError("NOT_INSPECTED", "solo se pueden derivar imagenes ya inspeccionadas; llame primero a inspect_image", False, 409)
    ex = ctx.conn.execute("SELECT queue_id, status FROM review_queue WHERE image_id=?", (a["image_id"],)).fetchone()
    if ex:                                                               # idempotente por image_id
        return dict(queue_id=ex["queue_id"], status=ex["status"], already_queued=True)
    qid = "Q-" + hashlib.sha1(f"{L}|{a['image_id']}".encode()).hexdigest()[:8]      # determinista: un caso por imagen y linea
    res = json.loads(insp["result_json"])
    ctx.conn.execute("INSERT INTO review_queue(queue_id,line_id,image_id,reason,model_decision,status,created_at,created_by,model_version) "
                     "VALUES(?,?,?,?,?,?,?,?,?)", (qid, L, a["image_id"], a["reason"], res.get("decision"), "open", audit.now_iso(),
                                                   ctx.ident.user_id, insp["model_version"]))
    return dict(queue_id=qid, status="open", already_queued=False, _audit=dict(reason=a["reason"][:200]))


def list_review_queue(ctx, a):
    q = "SELECT queue_id,image_id,reason,model_decision,status,created_at FROM review_queue WHERE line_id=?"
    p = [ctx.ident.line_id]
    if a.get("status"):
        q += " AND status=?"
        p.append(a["status"])
    rows = ctx.conn.execute(q + " ORDER BY created_at, queue_id LIMIT ?", p + [a.get("limit", 20)]).fetchall()
    return dict(cases=[dict(r) for r in rows], count=len(rows), _audit=dict(filter=a.get("status"), returned=len(rows)))


def confirm_review_label(ctx, a):
    L = ctx.ident.line_id
    q = ctx.conn.execute("SELECT * FROM review_queue WHERE queue_id=? AND line_id=?", (a["queue_id"], L)).fetchone()
    if not q:
        raise ToolError("NOT_FOUND", "caso no encontrado en su linea", False, 404)
    if not ctx.conn.execute("SELECT 1 FROM skus WHERE object_id=? AND active=1", (a["confirmed_class"],)).fetchone():
        raise ToolError("UNKNOWN_CLASS", "la clase no existe en el catalogo de SKU", False, 422)
    ex = ctx.conn.execute("SELECT * FROM labels_staging WHERE queue_id=?", (a["queue_id"],)).fetchone()
    ang = a.get("confirmed_angle")
    if ex:
        if ex["confirmed_class"] == a["confirmed_class"] and ex["confirmed_angle"] == ang:
            return dict(label_id=ex["label_id"], status=ex["status"], already_labeled=True)
        raise ToolError("CONFLICT", "el caso ya tiene otra etiqueta confirmada; no se sobrescribe", False, 409, label_id=ex["label_id"])
    if q["status"] != "open":
        raise ToolError("CASE_NOT_OPEN", f"el caso esta en estado {q['status']}", False, 409)
    lid = "L-" + uuid.uuid4().hex[:8]
    ctx.conn.execute("INSERT INTO labels_staging(label_id,queue_id,line_id,confirmed_class,confirmed_angle,labeled_by,model_version,created_at,status)"
                     " VALUES(?,?,?,?,?,?,?,?,?)", (lid, a["queue_id"], L, a["confirmed_class"], ang, ctx.ident.user_id, q["model_version"],
                                                    audit.now_iso(), "staged_for_retraining"))
    ctx.conn.execute("UPDATE review_queue SET status='labeled' WHERE queue_id=?", (a["queue_id"],))
    return dict(label_id=lid, status="staged_for_retraining", already_labeled=False,
                _audit=dict(confirmed_class=a["confirmed_class"], confirmed_angle=ang, rejected_by_model_version=q["model_version"]))


SPECS = [
    ToolSpec("inspect_image",
             "Valida la calidad de 1 a 5 imagenes ya cargadas, identifica el objeto, estima el angulo y devuelve la decision "
             "GRASP / RETRY_VIEW / REJECT con confianza calibrada, incertidumbre y vistas similares como evidencia. "
             "Si la calidad no alcanza, no se llama al modelo y se pide volver a capturar. Solo lectura.",
             {"type": "object", "additionalProperties": False, "required": ["image_ids"],
              "properties": {"image_ids": {"type": "array", "minItems": 1, "maxItems": 5, "uniqueItems": True, "items": ID},
                             "line_id": {"type": "string"}}},
             inspect_image),
    ToolSpec("queue_reject",
             "Deriva a la cola de revision humana una imagen ya inspeccionada (normalmente con decision REJECT) y devuelve su queue_id.",
             {"type": "object", "additionalProperties": False, "required": ["image_id", "reason"],
              "properties": {"image_id": ID, "reason": {"type": "string", "minLength": 3, "maxLength": 200}, "line_id": {"type": "string"}}},
             queue_reject, roles=("supervisor", "calidad"), write=True),
    ToolSpec("list_review_queue", "Lista hasta 20 casos de la cola de revision humana de su linea.",
             {"type": "object", "additionalProperties": False,
              "properties": {"status": {"type": "string", "enum": ["open", "labeled", "manual_angle_sent"]},
                             "limit": {"type": "integer", "minimum": 1, "maximum": 20}, "line_id": {"type": "string"}}},
             list_review_queue, roles=("supervisor", "calidad")),
    ToolSpec("confirm_review_label",
             "Registra la etiqueta (clase y angulo opcional) que confirmo una persona para un caso de la cola y la deja en staging de reentrenamiento.",
             {"type": "object", "additionalProperties": False, "required": ["queue_id", "confirmed_class"],
              "properties": {"queue_id": {"type": "string", "pattern": "^Q-[a-z0-9]{1,16}$"},
                             "confirmed_class": {"type": "integer", "minimum": 1, "maximum": 100000},
                             "confirmed_angle": {"type": "number", "minimum": 0, "exclusiveMaximum": 360}, "line_id": {"type": "string"}}},
             confirm_review_label, roles=("supervisor", "calidad"), write=True),
]


def make_server(**kw):
    return ToolServer("vision", SPECS, **kw)


if __name__ == "__main__":
    make_server().run()
