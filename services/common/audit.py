"""Auditoria de solo adicion con cadena de hashes (cada registro incluye el hash del anterior). Sin imagenes ni secretos."""
import hashlib
import json
from datetime import datetime, timezone


def _canon(o):
    return json.dumps(o, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append(conn, record):
    """Agrega un registro; la lectura del ultimo hash y la escritura ocurren en una transaccion (seguro entre procesos)."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute("SELECT hash FROM audit_log ORDER BY seq DESC LIMIT 1").fetchone()
        prev = row["hash"] if row else "0" * 64
        body = _canon(record)
        ts = now_iso()
        h = hashlib.sha256((prev + ts + body).encode()).hexdigest()
        conn.execute("INSERT INTO audit_log(ts, record_json, prev_hash, hash) VALUES(?,?,?,?)", (ts, body, prev, h))
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return h


def verify_chain(conn):
    """Devuelve (ok, n_registros, seq_del_primer_error)."""
    prev = "0" * 64
    n = 0
    for r in conn.execute("SELECT * FROM audit_log ORDER BY seq"):
        n += 1
        if r["prev_hash"] != prev or hashlib.sha256((prev + r["ts"] + r["record_json"]).encode()).hexdigest() != r["hash"]:
            return False, n, r["seq"]
        prev = r["hash"]
    return True, n, None
