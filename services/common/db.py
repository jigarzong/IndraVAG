"""Memoria operativa persistente (SQLite). Mismo esquema para el MVP local; en produccion seria una base relacional gestionada."""
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS images(image_id TEXT PRIMARY KEY, line_id TEXT NOT NULL, path TEXT NOT NULL, sha256 TEXT NOT NULL,
  uploaded_by TEXT, created_at TEXT, angle_deg REAL, quality_ok INTEGER DEFAULT 1, quality_flags TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS inspections(image_id TEXT PRIMARY KEY, line_id TEXT NOT NULL, result_json TEXT NOT NULL,
  model_version TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS review_queue(queue_id TEXT PRIMARY KEY, line_id TEXT NOT NULL, image_id TEXT NOT NULL UNIQUE,
  reason TEXT, model_decision TEXT, status TEXT NOT NULL DEFAULT 'open', created_at TEXT, created_by TEXT, model_version TEXT);
CREATE TABLE IF NOT EXISTS labels_staging(label_id TEXT PRIMARY KEY, queue_id TEXT NOT NULL UNIQUE, line_id TEXT NOT NULL,
  confirmed_class INTEGER NOT NULL, confirmed_angle REAL, labeled_by TEXT, model_version TEXT, created_at TEXT, status TEXT);
CREATE TABLE IF NOT EXISTS turns(line_id TEXT NOT NULL, turn_id TEXT NOT NULL, start_ts TEXT NOT NULL, end_ts TEXT NOT NULL,
  closed INTEGER NOT NULL, PRIMARY KEY(line_id, turn_id));
CREATE TABLE IF NOT EXISTS metrics(line_id TEXT NOT NULL, turn_id TEXT NOT NULL, hour_start TEXT NOT NULL, throughput INTEGER,
  rejects INTEGER, grasp_failures INTEGER, downtime_min REAL, PRIMARY KEY(line_id, hour_start));
CREATE TABLE IF NOT EXISTS camera_hourly(line_id TEXT NOT NULL, hour_start TEXT NOT NULL, sharpness REAL, exposure REAL,
  capture_failures INTEGER, PRIMARY KEY(line_id, hour_start));
CREATE TABLE IF NOT EXISTS camera_status(line_id TEXT PRIMARY KEY, last_calibration TEXT);
CREATE TABLE IF NOT EXISTS line_config(line_id TEXT PRIMARY KEY, active_model_version TEXT, rollback_model_version TEXT,
  confidence_threshold REAL, line_state TEXT);
CREATE TABLE IF NOT EXISTS config_changes(id INTEGER PRIMARY KEY AUTOINCREMENT, line_id TEXT, ts TEXT, kind TEXT, old TEXT,
  new TEXT, actor TEXT, request_id TEXT);
CREATE TABLE IF NOT EXISTS model_versions(version TEXT PRIMARY KEY, status TEXT, created_at TEXT, evidence_report_id TEXT, notes TEXT);
CREATE TABLE IF NOT EXISTS evidence_reports(report_id TEXT PRIMARY KEY, candidate_version TEXT, sku_name TEXT, protocol TEXT,
  acc_clase REAL, mae_circular REAL, pct_within_10 REAL, created_at TEXT, simulated INTEGER DEFAULT 1);
CREATE TABLE IF NOT EXISTS evidence_fixtures(sku_name TEXT, protocol TEXT, acc_clase REAL, mae_circular REAL, pct_within_10 REAL, PRIMARY KEY(sku_name, protocol));
CREATE TABLE IF NOT EXISTS skus(object_id INTEGER PRIMARY KEY, name TEXT, active INTEGER DEFAULT 1);
CREATE TABLE IF NOT EXISTS sku_staging(batch_id TEXT PRIMARY KEY, line_id TEXT, sku_name TEXT, n_views INTEGER, image_ids TEXT,
  fingerprint TEXT, status TEXT, missing_angles TEXT, created_by TEXT, created_at TEXT, UNIQUE(sku_name, fingerprint));
CREATE TABLE IF NOT EXISTS tickets(ticket_id TEXT PRIMARY KEY, line_id TEXT NOT NULL, equipment TEXT, priority TEXT, issue_desc TEXT,
  norm_key TEXT, status TEXT DEFAULT 'open', created_at TEXT, created_by TEXT);
CREATE TABLE IF NOT EXISTS shift_reports(line_id TEXT NOT NULL, turn_id TEXT NOT NULL, report_id TEXT, summary_json TEXT,
  created_at TEXT, PRIMARY KEY(line_id, turn_id));
CREATE TABLE IF NOT EXISTS sop_docs(doc_id TEXT PRIMARY KEY, title TEXT, equipment TEXT, text TEXT, version TEXT);
CREATE TABLE IF NOT EXISTS approval_requests(request_id TEXT PRIMARY KEY, line_id TEXT NOT NULL, tool TEXT NOT NULL,
  payload_json TEXT NOT NULL, payload_hash TEXT NOT NULL, requester TEXT, requester_role TEXT, status TEXT NOT NULL,
  context_json TEXT, created_at TEXT, expires_at TEXT, decided_by TEXT, decided_at TEXT, result_json TEXT, reason TEXT);
CREATE TABLE IF NOT EXISTS idempotency(key TEXT PRIMARY KEY, tool TEXT, result_json TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS robot_outbox(id INTEGER PRIMARY KEY AUTOINCREMENT, line_id TEXT, image_id TEXT, angle REAL,
  request_id TEXT, sent_at TEXT);
CREATE TABLE IF NOT EXISTS audit_log(seq INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, record_json TEXT NOT NULL,
  prev_hash TEXT NOT NULL, hash TEXT NOT NULL);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit_log BEGIN SELECT RAISE(ABORT, 'audit_log es de solo adicion'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit_log BEGIN SELECT RAISE(ABORT, 'audit_log es de solo adicion'); END;
"""


def connect(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(path), timeout=15, isolation_level=None, check_same_thread=False)   # el servidor MCP ejecuta las tools en un hilo; invoke() se serializa con un candado      # autocommit; las transacciones se abren explicitamente
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA busy_timeout=15000")
    return c


def init_schema(c):
    c.executescript(SCHEMA)
    return c
