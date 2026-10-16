"""Servidor MCP de Ingenieria ML: alta de SKU (staging), evaluacion y solicitud de promocion de version.

La evaluacion de `test_sku_model` es SIMULADA en el MVP (ver services/common/fixtures.py): reutiliza las metricas medidas del Reto 2
como valores de referencia, no reentrena ni evalua el SKU real. Lo que si es real: staging, validaciones, minimos de promocion y aprobacion.
"""
import hashlib
import json
import uuid

from services.common import audit
from services.common.toolkit import ToolError, ToolServer, ToolSpec


def _gaps(angles):
    a = sorted(x % 360 for x in angles)
    if not a:
        return 360.0
    return max([(a[(i + 1) % len(a)] - a[i]) % 360 or (360 if len(a) == 1 else 0) for i in range(len(a))])


def upload_new_sku_views(ctx, a):
    L, cfg = ctx.ident.line_id, ctx.cfg
    if ctx.conn.execute("SELECT 1 FROM skus WHERE lower(name)=lower(?)", (a["sku_name"],)).fetchone():
        raise ToolError("SKU_EXISTS", "ya existe un SKU con ese nombre en el catalogo", False, 409)
    rows, missing = [], []
    for i in a["image_ids"]:
        r = ctx.conn.execute("SELECT * FROM images WHERE image_id=? AND line_id=?", (i, L)).fetchone()
        (rows if r else missing).append(r or i)
    if missing:
        raise ToolError("IMAGE_NOT_FOUND", "algunas imagenes no existen en su linea", False, 404, missing_ids=missing[:20])
    n = len(rows)
    bad = [r["image_id"] for r in rows if not r["quality_ok"]]
    angles = [r["angle_deg"] for r in rows if r["angle_deg"] is not None]
    bins = [b for b in range(0, 360, 10) if not any(min((x - b) % 360, (b - x) % 360) <= 5 for x in angles)]
    gap = _gaps(angles)
    problems = []
    if n < cfg.sku_min_views:
        problems.append(f"vistas insuficientes ({n} de {cfg.sku_min_views})")
    if bad:
        problems.append(f"{len(bad)} imagenes no pasan calidad")
    if len(angles) < n:
        problems.append("hay imagenes sin angulo de captura")
    if angles and gap > cfg.sku_max_angle_gap_deg:
        problems.append(f"hueco angular de {gap:.0f} grados (maximo {cfg.sku_max_angle_gap_deg:g})")
    status = "validated" if not problems else "insufficient"
    fp = hashlib.sha256("|".join(sorted(r["sha256"] for r in rows)).encode()).hexdigest()[:16]
    ex = ctx.conn.execute("SELECT batch_id FROM sku_staging WHERE sku_name=? AND fingerprint=?", (a["sku_name"], fp)).fetchone()
    bid = ex["batch_id"] if ex else "B-" + uuid.uuid4().hex[:8]
    if not ex:                                                      # escribe en staging, nunca en el modelo productivo
        ctx.conn.execute("INSERT INTO sku_staging(batch_id,line_id,sku_name,n_views,image_ids,fingerprint,status,missing_angles,created_by,created_at)"
                         " VALUES(?,?,?,?,?,?,?,?,?,?)", (bid, L, a["sku_name"], n, json.dumps(a["image_ids"]), fp, status, json.dumps(bins),
                                                          ctx.ident.user_id, audit.now_iso()))
    return dict(batch_id=bid, status=status, views_received=n, views_required=cfg.sku_min_views, quality_failures=bad[:20],
                missing_angles_deg=bins[:36], max_angle_gap_deg=round(gap, 1), problems=problems, already_staged=bool(ex),
                _audit=dict(sku_name=a["sku_name"], batch_fingerprint=fp, views=n))


def test_sku_model(ctx, a):
    L = ctx.ident.line_id
    b = ctx.conn.execute("SELECT * FROM sku_staging WHERE line_id=? AND sku_name=? AND status='validated' ORDER BY created_at DESC LIMIT 1",
                         (L, a["sku_name"])).fetchone()
    if not b:
        raise ToolError("NOT_VALIDATED", "el SKU no tiene un lote validado en staging; suba y valide las vistas primero", False, 409)
    fx = ctx.conn.execute("SELECT * FROM evidence_fixtures WHERE sku_name=? AND protocol=?", (a["sku_name"], a["protocol"])).fetchone()
    if not fx:
        raise ToolError("NO_EVIDENCE_AVAILABLE", "el MVP solo tiene evaluacion simulada para los SKU de demostracion", False, 501)
    active = ctx.conn.execute("SELECT active_model_version FROM line_config WHERE line_id=?", (L,)).fetchone()[0]
    cand = f"{active}+{a['sku_name']}"
    rid = "ER-" + hashlib.sha256(f"{cand}|{a['protocol']}".encode()).hexdigest()[:8]
    ctx.conn.execute("INSERT OR IGNORE INTO model_versions(version,status,created_at,notes) VALUES(?,?,?,?)",
                     (cand, "candidate", audit.now_iso(), f"candidata con SKU {a['sku_name']} (simulada)"))
    ctx.conn.execute("INSERT OR REPLACE INTO evidence_reports(report_id,candidate_version,sku_name,protocol,acc_clase,mae_circular,pct_within_10,created_at,simulated)"
                     " VALUES(?,?,?,?,?,?,?,?,1)", (rid, cand, a["sku_name"], a["protocol"], fx["acc_clase"], fx["mae_circular"], fx["pct_within_10"], audit.now_iso()))
    return dict(evidence_report_id=rid, candidate_version=cand, protocol=a["protocol"], acc_clase=fx["acc_clase"],
                mae_circular_deg=fx["mae_circular"], pct_within_10=fx["pct_within_10"], simulated=True,
                note="evaluacion SIMULADA: valores de referencia del Reto 2, no una evaluacion real del SKU",
                _audit=dict(sku_name=a["sku_name"], protocol=a["protocol"], candidate_version=cand))


def promote_model_version(ctx, a):
    L, cfg = ctx.ident.line_id, ctx.cfg
    ev = ctx.conn.execute("SELECT * FROM evidence_reports WHERE report_id=?", (a["evidence_report_id"],)).fetchone()
    if not ev or ev["candidate_version"] != a["candidate_version"]:
        raise ToolError("EVIDENCE_NOT_FOUND", "la evidencia no existe o no corresponde a la version candidata", False, 404)
    cand = ctx.conn.execute("SELECT * FROM model_versions WHERE version=?", (a["candidate_version"],)).fetchone()
    if not cand or cand["status"] != "candidate":
        raise ToolError("NOT_A_CANDIDATE", "la version no esta como candidata", False, 409)
    # los minimos los evalua el middleware, no el LLM
    fails = []
    if ev["acc_clase"] is None or ev["acc_clase"] < cfg.promotion_min_acc_clase:
        fails.append(f"exactitud de clase {ev['acc_clase']} < {cfg.promotion_min_acc_clase}")
    if ev["pct_within_10"] is None or ev["pct_within_10"] < cfg.promotion_min_pct_within_10 * 100:
        fails.append(f"dentro de 10 grados {ev['pct_within_10']} < {cfg.promotion_min_pct_within_10 * 100:g}")
    if fails:
        raise ToolError("EVIDENCE_BELOW_MINIMUMS", "la evidencia no cumple los minimos de promocion; no se crea la solicitud", False, 422,
                        failures=fails, _audit=dict(candidate_version=a["candidate_version"], evidence_report_id=a["evidence_report_id"]))
    active = ctx.conn.execute("SELECT active_model_version FROM line_config WHERE line_id=?", (L,)).fetchone()[0]
    rid, created = ctx.approvals.create(ctx.ident, "promote_model_version",
                                        dict(candidate_version=a["candidate_version"], evidence_report_id=a["evidence_report_id"], reason=a["reason"]),
                                        dict(active=active, candidate_status=cand["status"]))
    return dict(status="pending_human_approval", request_id=rid, new_request=created,
                note="debe aprobar otro usuario autorizado; la aprobacion caduca si cambia la version candidata o la activa",
                _audit=dict(request_id=rid, candidate_version=a["candidate_version"], evidence_report_id=a["evidence_report_id"],
                            active_version=active, rollback_version=active))


SPECS = [
    ToolSpec("upload_new_sku_views",
             "Valida un lote de vistas de un SKU nuevo (cantidad minima, calidad de cada imagen y cobertura de angulos) y lo deja en staging. Nunca toca el modelo productivo.",
             {"type": "object", "additionalProperties": False, "required": ["sku_name", "image_ids"],
              "properties": {"sku_name": {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9_-]{1,39}$"},
                             "image_ids": {"type": "array", "minItems": 1, "maxItems": 200, "uniqueItems": True,
                                           "items": {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,64}$"}}, "line_id": {"type": "string"}}},
             upload_new_sku_views, roles=("ingeniero_ml",), write=True),
    ToolSpec("test_sku_model",
             "Evalua en staging el modelo candidato con el SKU nuevo segun el protocolo A, B, C o D y devuelve exactitud, error angular circular medio, % dentro de 10 grados y un evidence_report_id.",
             {"type": "object", "additionalProperties": False, "required": ["sku_name", "protocol"],
              "properties": {"sku_name": {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9_-]{1,39}$"},
                             "protocol": {"type": "string", "enum": ["A", "B", "C", "D"]}, "line_id": {"type": "string"}}},
             test_sku_model, roles=("ingeniero_ml",)),
    ToolSpec("promote_model_version",
             "Prepara la promocion de una version candidata a produccion. NO la aplica: el middleware valida la evidencia contra los minimos y la solicitud queda pendiente de aprobacion de otro usuario.",
             {"type": "object", "additionalProperties": False, "required": ["candidate_version", "evidence_report_id", "reason"],
              "properties": {"candidate_version": {"type": "string", "minLength": 3, "maxLength": 80},
                             "evidence_report_id": {"type": "string", "pattern": "^ER-[a-f0-9]{8}$"},
                             "reason": {"type": "string", "minLength": 5, "maxLength": 300}, "line_id": {"type": "string"}}},
             promote_model_version, roles=("ingeniero_ml",), write=True, risk=True),
]


def make_server(**kw):
    return ToolServer("ingenieria_ml", SPECS, **kw)


if __name__ == "__main__":
    make_server().run()
