"""Aprobacion humana (HITL) de operaciones de riesgo.

El LLM solo PREPARA una solicitud (queda `pending_human_approval`). La aprobacion la hace una persona por la API/UI de
aprobaciones, que NO es una tool MCP: ningun resultado de tool contiene el token, asi que el modelo no puede generarlo ni
reenviarlo. El efecto solo se aplica si el ejecutor verifica el token HMAC emitido en la aprobacion.
"""
import hashlib
import hmac
import json
import uuid
from datetime import datetime, timedelta, timezone

from . import audit

# quien puede aprobar cada tool de riesgo (decision de diseno del MVP; ajustable a la politica real de la planta)
APPROVER_ROLES = {
    "change_confidence_threshold": ("supervisor",),
    "control_line_state": ("supervisor",),
    "send_manual_angle": ("supervisor",),
    "promote_model_version": ("ingeniero_ml", "supervisor"),
}
REQUIRES_DISTINCT_APPROVER = {"promote_model_version"}


class ApprovalError(Exception):
    def __init__(self, code, message, status=403):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def _canon(o):
    return json.dumps(o, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def _now():
    return datetime.now(timezone.utc)


class ApprovalService:
    def __init__(self, conn, cfg):
        self.conn, self.cfg = conn, cfg
        self.executors = {}

    # ---- creacion (la llama la tool, en nombre del LLM)
    def create(self, ident, tool, payload, context):
        ph = hashlib.sha256(_canon([tool, payload, ident.line_id]).encode()).hexdigest()
        row = self.conn.execute("SELECT request_id FROM approval_requests WHERE payload_hash=? AND status='pending' AND expires_at>?",
                                (ph, _now().isoformat(timespec="seconds"))).fetchone()
        if row:
            return row["request_id"], False
        rid = "AR-" + uuid.uuid4().hex[:10]
        exp = (_now() + timedelta(minutes=self.cfg.approval_ttl_min)).isoformat(timespec="seconds")
        self.conn.execute("INSERT INTO approval_requests(request_id,line_id,tool,payload_json,payload_hash,requester,requester_role,status,"
                          "context_json,created_at,expires_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                          (rid, ident.line_id, tool, _canon(payload), ph, ident.user_id, ident.role, "pending", _canon(context),
                           audit.now_iso(), exp))
        return rid, True

    # ---- lo que usa la UI (no es una tool)
    def list_pending(self, approver):
        out = []
        for r in self.conn.execute("SELECT * FROM approval_requests WHERE line_id=? AND status='pending' AND expires_at>? ORDER BY created_at",
                                   (approver.line_id, _now().isoformat(timespec="seconds"))):
            if approver.role in APPROVER_ROLES.get(r["tool"], ()):
                out.append(dict(request_id=r["request_id"], tool=r["tool"], payload=json.loads(r["payload_json"]),
                                requester=r["requester"], created_at=r["created_at"], expires_at=r["expires_at"]))
        return out

    def _token(self, request_id, payload_hash, approver_id, ts):
        return hmac.new(self.cfg.approval_secret.encode(), f"{request_id}|{payload_hash}|{approver_id}|{ts}".encode(), hashlib.sha256).hexdigest()

    def verify_token(self, req, approver_id, ts, token):
        return hmac.compare_digest(self._token(req["request_id"], req["payload_hash"], approver_id, ts), token or "")

    def _load(self, request_id, approver):
        r = self.conn.execute("SELECT * FROM approval_requests WHERE request_id=?", (request_id,)).fetchone()
        if not r or r["line_id"] != approver.line_id:          # sin pistas sobre solicitudes de otras lineas
            raise ApprovalError("NOT_FOUND", "solicitud no encontrada", 404)
        return r

    def _expire(self, r, why):
        self.conn.execute("UPDATE approval_requests SET status='expired', reason=? WHERE request_id=?", (why, r["request_id"]))
        audit.append(self.conn, dict(event="approval_expired", request_id=r["request_id"], tool=r["tool"], reason=why))

    def approve(self, approver, request_id):
        if not self.cfg.approval_secret:
            raise ApprovalError("CONFIG", "APPROVAL_SECRET no configurado", 500)
        r = self._load(request_id, approver)
        if approver.role not in APPROVER_ROLES.get(r["tool"], ()):
            raise ApprovalError("FORBIDDEN", "su rol no puede aprobar esta operacion")
        if r["status"] != "pending":
            raise ApprovalError("NOT_PENDING", f"la solicitud esta en estado {r['status']}", 409)
        if r["expires_at"] <= _now().isoformat(timespec="seconds"):
            self._expire(r, "TTL")
            raise ApprovalError("EXPIRED", "la aprobacion caduco", 409)
        if r["tool"] in REQUIRES_DISTINCT_APPROVER and approver.user_id == r["requester"]:
            raise ApprovalError("SAME_USER", "el aprobador debe ser distinto del solicitante")
        ctx_now = self.executors[r["tool"]]["context"](self.conn, r["line_id"], json.loads(r["payload_json"]))
        if _canon(ctx_now) != r["context_json"]:               # cambio la version candidata/activa o el umbral desde la solicitud
            self._expire(r, "CONTEXT_CHANGED")
            raise ApprovalError("CONTEXT_CHANGED", "el contexto cambio desde la solicitud; debe solicitarse de nuevo", 409)
        ts = audit.now_iso()
        token = self._token(r["request_id"], r["payload_hash"], approver.user_id, ts)
        res = self.executors[r["tool"]]["apply"](self, r, approver, ts, token)
        self.conn.execute("UPDATE approval_requests SET status='executed', decided_by=?, decided_at=?, result_json=? WHERE request_id=?",
                          (approver.user_id, ts, _canon(res), r["request_id"]))
        audit.append(self.conn, dict(event="approval", request_id=r["request_id"], tool=r["tool"], line=r["line_id"],
                                     requester=r["requester"], approver=approver.user_id, approver_role=approver.role,
                                     payload=json.loads(r["payload_json"]), result=res))
        return res

    def reject(self, approver, request_id, note=""):
        r = self._load(request_id, approver)
        if approver.role not in APPROVER_ROLES.get(r["tool"], ()):
            raise ApprovalError("FORBIDDEN", "su rol no puede rechazar esta operacion")
        if r["status"] != "pending":
            raise ApprovalError("NOT_PENDING", f"la solicitud esta en estado {r['status']}", 409)
        self.conn.execute("UPDATE approval_requests SET status='rejected', decided_by=?, decided_at=?, reason=? WHERE request_id=?",
                          (approver.user_id, audit.now_iso(), note[:200], request_id))
        audit.append(self.conn, dict(event="approval_rejected", request_id=request_id, tool=r["tool"], approver=approver.user_id))
        return dict(request_id=request_id, status="rejected")
