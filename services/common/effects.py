"""Efectos de las operaciones de riesgo. Solo se ejecutan tras una aprobacion humana valida (token HMAC verificado)."""
import json

from . import audit


def _line(conn, line_id):
    return conn.execute("SELECT * FROM line_config WHERE line_id=?", (line_id,)).fetchone()


def _need_token(svc, req, approver, ts, token):
    if not svc.verify_token(dict(request_id=req["request_id"], payload_hash=req["payload_hash"]), approver.user_id, ts, token):
        raise PermissionError("token de aprobacion invalido")


def _log_change(conn, line_id, kind, old, new, actor, request_id):
    conn.execute("INSERT INTO config_changes(line_id,ts,kind,old,new,actor,request_id) VALUES(?,?,?,?,?,?,?)",
                 (line_id, audit.now_iso(), kind, str(old), str(new), actor, request_id))


# ---- umbral de confianza
def ctx_threshold(conn, line_id, payload):
    return {"current": _line(conn, line_id)["confidence_threshold"]}


def apply_threshold(svc, req, approver, ts, token):
    _need_token(svc, req, approver, ts, token)
    p, c = json.loads(req["payload_json"]), svc.conn
    old = _line(c, req["line_id"])["confidence_threshold"]
    c.execute("UPDATE line_config SET confidence_threshold=? WHERE line_id=?", (p["new_threshold"], req["line_id"]))
    _log_change(c, req["line_id"], "threshold", old, p["new_threshold"], approver.user_id, req["request_id"])
    return dict(applied=True, old=old, new=p["new_threshold"])


# ---- pausar / reanudar la linea
def ctx_state(conn, line_id, payload):
    return {"state": _line(conn, line_id)["line_state"]}


def apply_state(svc, req, approver, ts, token):
    _need_token(svc, req, approver, ts, token)
    p, c = json.loads(req["payload_json"]), svc.conn
    new = "paused" if p["action"] == "pause" else "running"
    old = _line(c, req["line_id"])["line_state"]
    c.execute("UPDATE line_config SET line_state=? WHERE line_id=?", (new, req["line_id"]))
    _log_change(c, req["line_id"], "line_state", old, new, approver.user_id, req["request_id"])
    return dict(applied=True, old=old, new=new)


# ---- angulo manual al robot (el robot es simulado en el MVP: se registra en robot_outbox)
def ctx_angle(conn, line_id, payload):
    q = conn.execute("SELECT status FROM review_queue WHERE line_id=? AND image_id=?", (line_id, payload["image_id"])).fetchone()
    return {"queue_status": q["status"] if q else None}


def apply_angle(svc, req, approver, ts, token):
    _need_token(svc, req, approver, ts, token)
    p, c = json.loads(req["payload_json"]), svc.conn
    c.execute("INSERT INTO robot_outbox(line_id,image_id,angle,request_id,sent_at) VALUES(?,?,?,?,?)",
              (req["line_id"], p["image_id"], p["manual_angle"], req["request_id"], ts))
    c.execute("UPDATE review_queue SET status='manual_angle_sent' WHERE line_id=? AND image_id=?", (req["line_id"], p["image_id"]))
    return dict(applied=True, sent_to_robot=True, simulated_robot=True, angle=p["manual_angle"])


# ---- promocion de version de modelo (solo registro de versiones; el servicio de vision sirve el artefacto de la version activa)
def ctx_promote(conn, line_id, payload):
    cand = conn.execute("SELECT status FROM model_versions WHERE version=?", (payload["candidate_version"],)).fetchone()
    return {"active": _line(conn, line_id)["active_model_version"], "candidate_status": cand["status"] if cand else None}


def apply_promote(svc, req, approver, ts, token):
    _need_token(svc, req, approver, ts, token)
    p, c = json.loads(req["payload_json"]), svc.conn
    old = _line(c, req["line_id"])["active_model_version"]
    c.execute("BEGIN IMMEDIATE")
    try:
        c.execute("UPDATE model_versions SET status='rollback' WHERE version=?", (old,))
        c.execute("UPDATE model_versions SET status='active' WHERE version=?", (p["candidate_version"],))
        c.execute("UPDATE line_config SET active_model_version=?, rollback_model_version=? WHERE line_id=?",
                  (p["candidate_version"], old, req["line_id"]))
        c.execute("COMMIT")
    except Exception:
        c.execute("ROLLBACK")
        raise
    _log_change(c, req["line_id"], "model_version", old, p["candidate_version"], approver.user_id, req["request_id"])
    return dict(applied=True, active_version=p["candidate_version"], rollback_version=old)


EXECUTORS = {
    "change_confidence_threshold": dict(context=ctx_threshold, apply=apply_threshold),
    "control_line_state": dict(context=ctx_state, apply=apply_state),
    "send_manual_angle": dict(context=ctx_angle, apply=apply_angle),
    "promote_model_version": dict(context=ctx_promote, apply=apply_promote),
}
