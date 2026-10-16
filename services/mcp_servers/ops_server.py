"""Servidor MCP de Operaciones: metricas, contexto de linea, reporte de turno y las tres operaciones de riesgo (con aprobacion humana).

Los datos son SIMULADOS (services/common/fixtures.py): el MVP no esta conectado a una planta real.
"""
import hashlib
import json
from datetime import datetime, timedelta

from services.common import audit
from services.common.toolkit import ToolError, ToolServer, ToolSpec

TURN_RE = r"^(\d{4}-\d{2}-\d{2}-)?T[123]$"
METRICS = {"throughput": ("throughput", "picks"), "rejects": ("rejects", "casos"), "grasp_failures": ("grasp_failures", "fallos"),
           "downtime": ("downtime_min", "min")}


def _turn(ctx, turn_id):
    full = turn_id if len(turn_id) > 3 else f"{ctx.cfg.demo_today}-{turn_id}"
    t = ctx.conn.execute("SELECT * FROM turns WHERE line_id=? AND turn_id=?", (ctx.ident.line_id, full)).fetchone()
    if not t:
        raise ToolError("TURN_NOT_FOUND", f"el turno {full} no existe para su linea", False, 404)
    return t


def _prev_turn(ctx, t):
    return ctx.conn.execute("SELECT * FROM turns WHERE line_id=? AND start_ts<? ORDER BY start_ts DESC LIMIT 1", (t["line_id"], t["start_ts"])).fetchone()


def _rows(ctx, t):
    return ctx.conn.execute("SELECT * FROM metrics WHERE line_id=? AND turn_id=? ORDER BY hour_start", (t["line_id"], t["turn_id"])).fetchall()


def query_metrics_turn(ctx, a):
    col, unit = METRICS[a["metric"]]
    t = _turn(ctx, a["turn_id"])
    rows = _rows(ctx, t)
    res = dict(turn_id=t["turn_id"], metric=a["metric"], unit=unit, interval=a["interval"], turn_closed=bool(t["closed"]),
               hours_with_data=len(rows))
    if a["interval"] == "hour":
        res["series"] = [dict(hour_start=r["hour_start"], value=r[col]) for r in rows]
    total = round(sum(r[col] for r in rows), 1)
    res["value"] = total
    p = _prev_turn(ctx, t)
    if p:
        pv = round(sum(r[col] for r in _rows(ctx, p)), 1)
        res["previous_turn"] = dict(turn_id=p["turn_id"], value=pv,
                                    change_pct=round((total - pv) / pv * 100, 1) if pv else None,
                                    comparable=bool(p["closed"]) and len(rows) == len(_rows(ctx, p)))
    res["_audit"] = dict(turn_id=t["turn_id"], metric=a["metric"], interval=a["interval"])
    return res


def _fmt(ts):
    return ts.replace(" ", "T")


def get_line_context(ctx, a):
    L = ctx.ident.line_id
    w = a["window"]
    if w["kind"] == "turn":
        t = _turn(ctx, w["turn_id"])
        start, end = t["start_ts"], t["end_ts"]
    else:
        start, end = _fmt(w["start"]), _fmt(w["end"])
        if datetime.fromisoformat(end) - datetime.fromisoformat(start) > timedelta(hours=24) or end <= start:
            raise ToolError("VALIDATION_ERROR", "la ventana debe ser positiva y de maximo 24 horas", False, 422, field="window")
    cam = ctx.conn.execute("SELECT * FROM camera_status WHERE line_id=?", (L,)).fetchone()
    series = ctx.conn.execute("SELECT hour_start,sharpness,exposure,capture_failures FROM camera_hourly WHERE line_id=? AND hour_start>=? "
                              "AND hour_start<? ORDER BY hour_start", (L, start, end)).fetchall()
    cfg = ctx.conn.execute("SELECT * FROM line_config WHERE line_id=?", (L,)).fetchone()
    ch = ctx.conn.execute("SELECT ts,kind,old,new FROM config_changes WHERE line_id=? AND ts>=? AND ts<? ORDER BY ts", (L, start, end)).fetchall()
    n = max(len(series), 1)
    return dict(window=dict(start=start, end=end),
                camera=dict(last_calibration=cam["last_calibration"] if cam else None,
                            mean_sharpness=round(sum(r["sharpness"] for r in series) / n, 1) if series else None,
                            mean_exposure=round(sum(r["exposure"] for r in series) / n, 1) if series else None,
                            capture_failures=sum(r["capture_failures"] for r in series),
                            hourly=[dict(r) for r in series]),
                model=dict(active_version=cfg["active_model_version"], confidence_threshold=cfg["confidence_threshold"],
                           line_state=cfg["line_state"]),
                changes_in_window=[dict(r) for r in ch], _audit=dict(window=[start, end]))


def generate_shift_report(ctx, a):
    L = ctx.ident.line_id
    t = _turn(ctx, a["turn_id"])
    if not t["closed"]:
        raise ToolError("TURN_NOT_CLOSED", "el turno aun no esta cerrado; el reporte se genera al cierre", False, 409)
    rows = _rows(ctx, t)
    tot = {k: sum(r[k] for r in rows) for k in ("throughput", "rejects", "grasp_failures", "downtime_min")}
    tot["reject_rate_pct"] = round(tot["rejects"] / max(tot["throughput"], 1) * 100, 2)
    inc = [dict(ts=r["ts"], kind=r["kind"]) for r in ctx.conn.execute(
        "SELECT ts,kind FROM config_changes WHERE line_id=? AND ts>=? AND ts<? ORDER BY ts", (L, t["start_ts"], t["end_ts"]))]
    tk = ctx.conn.execute("SELECT ticket_id,priority,status FROM tickets WHERE line_id=? AND status='open' ORDER BY ticket_id", (L,)).fetchall()
    q = ctx.conn.execute("SELECT queue_id FROM review_queue WHERE line_id=? AND status='open' ORDER BY queue_id", (L,)).fetchall()
    summary = dict(turn_id=t["turn_id"], totals=tot, incidents=inc, open_tickets=[dict(r) for r in tk],
                   pending_review_cases=dict(count=len(q), ids=[r["queue_id"] for r in q][:20]))
    rid = "SR-" + hashlib.sha256(f"{L}|{t['turn_id']}".encode()).hexdigest()[:8]
    prev = ctx.conn.execute("SELECT 1 FROM shift_reports WHERE line_id=? AND turn_id=?", (L, t["turn_id"])).fetchone()
    ctx.conn.execute("INSERT OR REPLACE INTO shift_reports(line_id,turn_id,report_id,summary_json,created_at) VALUES(?,?,?,?,?)",
                     (L, t["turn_id"], rid, json.dumps(summary, ensure_ascii=False), audit.now_iso()))
    return dict(report_id=rid, summary=summary, replaced_previous=bool(prev), _audit=dict(turn_id=t["turn_id"], replaced=bool(prev)))


def _pending(ctx, tool, payload, context, extra=None):
    rid, created = ctx.approvals.create(ctx.ident, tool, payload, context)
    return dict(status="pending_human_approval", request_id=rid, new_request=created,
                note="una persona autorizada debe aprobar en la interfaz; el asistente no puede aprobar ni ver el token",
                _audit=dict(payload=payload, request_id=rid, **(extra or {})))


def change_confidence_threshold(ctx, a):
    cur = ctx.conn.execute("SELECT confidence_threshold FROM line_config WHERE line_id=?", (ctx.ident.line_id,)).fetchone()[0]
    if abs(cur - a["new_threshold"]) < 1e-9:
        return dict(status="no_change", current=cur)
    return _pending(ctx, "change_confidence_threshold", dict(new_threshold=a["new_threshold"], reason=a["reason"]), dict(current=cur))


def control_line_state(ctx, a):
    st = ctx.conn.execute("SELECT line_state FROM line_config WHERE line_id=?", (ctx.ident.line_id,)).fetchone()[0]
    target = "paused" if a["action"] == "pause" else "running"
    if st == target:
        return dict(status="no_effect", line_state=st, note=f"la linea ya esta en estado {st}")
    return _pending(ctx, "control_line_state", dict(action=a["action"], reason=a["reason"]), dict(state=st))


def send_manual_angle(ctx, a):
    q = ctx.conn.execute("SELECT status FROM review_queue WHERE line_id=? AND image_id=?", (ctx.ident.line_id, a["image_id"])).fetchone()
    if not q:
        raise ToolError("NOT_IN_REVIEW_QUEUE", "el caso debe estar en la cola de revision de su linea", False, 409)
    if q["status"] not in ("open", "labeled"):
        raise ToolError("CASE_NOT_ACTIONABLE", f"el caso esta en estado {q['status']}", False, 409)
    return _pending(ctx, "send_manual_angle", dict(image_id=a["image_id"], manual_angle=a["manual_angle"], reason=a["reason"]),
                    dict(queue_status=q["status"]))


LINE = {"line_id": {"type": "string"}}
SPECS = [
    ToolSpec("query_metrics_turn",
             "Consulta una metrica de un turno (throughput, rejects, grasp_failures, downtime), total o por hora, comparada con el turno anterior. Solo lectura.",
             {"type": "object", "additionalProperties": False, "required": ["turn_id", "metric", "interval"],
              "properties": {"turn_id": {"type": "string", "pattern": TURN_RE, "description": "AAAA-MM-DD-T1..T3, o T1..T3 para el dia de hoy"},
                             "metric": {"type": "string", "enum": list(METRICS)}, "interval": {"type": "string", "enum": ["turn", "hour"]}, **LINE}},
             query_metrics_turn),
    ToolSpec("get_line_context",
             "Estado de la camara (calibracion, nitidez, exposicion, fallos), version de modelo y umbral vigentes, y cambios de version, umbral o estado dentro de una ventana de maximo 24 h. Solo lectura.",
             {"type": "object", "additionalProperties": False, "required": ["window"],
              "properties": {"window": {"oneOf": [
                  {"type": "object", "additionalProperties": False, "required": ["kind", "turn_id"],
                   "properties": {"kind": {"const": "turn"}, "turn_id": {"type": "string", "pattern": TURN_RE}}},
                  {"type": "object", "additionalProperties": False, "required": ["kind", "start", "end"],
                   "properties": {"kind": {"const": "range"}, "start": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?$"},
                                  "end": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?$"}}}]}, **LINE}},
             get_line_context),
    ToolSpec("generate_shift_report", "Genera (o reemplaza) el reporte de entrega de un turno ya cerrado: metricas, incidentes, tickets abiertos y casos pendientes.",
             {"type": "object", "additionalProperties": False, "required": ["turn_id"],
              "properties": {"turn_id": {"type": "string", "pattern": TURN_RE}, **LINE}},
             generate_shift_report, roles=("supervisor",), write=True),
    ToolSpec("change_confidence_threshold",
             "Prepara el cambio del umbral de confianza de la linea. NO lo aplica: queda pendiente de aprobacion humana (devuelve request_id).",
             {"type": "object", "additionalProperties": False, "required": ["new_threshold", "reason"],
              "properties": {"new_threshold": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
                             "reason": {"type": "string", "minLength": 5, "maxLength": 300}, **LINE}},
             change_confidence_threshold, roles=("supervisor",), write=True, risk=True),
    ToolSpec("control_line_state",
             "Prepara pausar o reanudar la linea. NO lo aplica: queda pendiente de aprobacion humana (devuelve request_id).",
             {"type": "object", "additionalProperties": False, "required": ["action", "reason"],
              "properties": {"action": {"type": "string", "enum": ["pause", "resume"]},
                             "reason": {"type": "string", "minLength": 5, "maxLength": 300}, **LINE}},
             control_line_state, roles=("supervisor",), write=True, risk=True),
    ToolSpec("send_manual_angle",
             "Prepara el envio de un angulo manual al robot para un caso de la cola de revision. NO lo envia: requiere aprobacion humana (devuelve request_id).",
             {"type": "object", "additionalProperties": False, "required": ["image_id", "manual_angle", "reason"],
              "properties": {"image_id": {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,64}$"},
                             "manual_angle": {"type": "number", "minimum": 0, "exclusiveMaximum": 360},
                             "reason": {"type": "string", "minLength": 5, "maxLength": 300}, **LINE}},
             send_manual_angle, roles=("supervisor",), write=True, risk=True),
]


def make_server(**kw):
    return ToolServer("operaciones", SPECS, **kw)


if __name__ == "__main__":
    make_server().run()
