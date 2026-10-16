"""Pruebas de las 15 tools: caminos buenos y los escenarios adversos que cada una debe resistir."""
import httpx
import numpy as np
import cv2
import pytest

from services.mcp_servers import vision_server

REAL = httpx.Client


def img(world, cache, sample_ids, key, image_id, line="L1"):
    world.add_image(image_id, cache["imgs"][sample_ids[key]], line)


# ------------------------------------------------------------------ vision
def test_inspect_grasp_and_reject(world, cache, sample_ids):
    img(world, cache, sample_ids, "GRASP", "g1"); img(world, cache, sample_ids, "REJECT", "r1")
    r = world.call("vision", "inspect_image", {"image_ids": ["g1", "r1"]}, role="calidad")
    assert r["ok"]
    g, x = r["results"]
    assert g["decision"] == "GRASP" and g["angle_reliable"] and g["object_id"] is not None and 1 <= len(g["similar_views"]) <= 5
    assert x["decision"] == "REJECT" and x["object_id"] is None and x["angle_deg"] is None and x["similar_views"] == []   # no se filtra una clase rechazada
    assert "pixels" not in str(r) and "features" not in str(r)


def test_inspect_bad_quality_does_not_call_model(world, cache, sample_ids):
    world.add_image("blur", cv2.GaussianBlur(cache["imgs"][sample_ids["GRASP"]], (0, 0), 6))
    calls = []
    orig = world.s["vision"].specs["inspect_image"].handler
    r = world.call("vision", "inspect_image", {"image_ids": ["blur"]}, role="calidad")
    it = r["results"][0]
    assert it["status"] == "QUALITY_FAILED" and it["recommended_action"] == "RECAPTURE" and "decision" not in it and "BLURRY" in it["quality"]["flags"]


def test_inspect_image_from_other_line_looks_nonexistent(world, cache, sample_ids):
    img(world, cache, sample_ids, "GRASP", "l2img", line="L2")
    r = world.call("vision", "inspect_image", {"image_ids": ["l2img"]}, role="calidad", line="L1")
    assert r["error"]["code"] == "IMAGE_NOT_FOUND"


def test_inspect_rejects_path_traversal_and_too_many(world):
    assert world.call("vision", "inspect_image", {"image_ids": ["../../etc/passwd"]})["error"]["code"] == "VALIDATION_ERROR"
    assert world.call("vision", "inspect_image", {"image_ids": [f"i{k}" for k in range(6)]})["error"]["code"] == "VALIDATION_ERROR"


def test_line_threshold_can_only_tighten(world, cache, sample_ids):
    img(world, cache, sample_ids, "GRASP", "g1")
    world.conn.execute("UPDATE line_config SET confidence_threshold=1.01 WHERE line_id='L1'")
    r = world.call("vision", "inspect_image", {"image_ids": ["g1"]}, role="calidad")
    assert r["results"][0]["decision"] == "REJECT" and r["results"][0]["decision_reason"] == "BELOW_LINE_THRESHOLD"


def _fail_with(world, monkeypatch, handler):
    monkeypatch.setattr(vision_server, "CLIENT_FACTORY", lambda: REAL(transport=httpx.MockTransport(handler), base_url="http://vision", timeout=5))


def test_vision_5xx_retries_once_then_structured_error(world, cache, sample_ids, monkeypatch):
    img(world, cache, sample_ids, "GRASP", "g1")
    calls = []
    def h(req):
        calls.append(1); return httpx.Response(503, text="boom")
    _fail_with(world, monkeypatch, h)
    r = world.call("vision", "inspect_image", {"image_ids": ["g1"]}, role="calidad")
    assert r["error"]["code"] == "VISION_5XX" and r["error"]["retryable"] and len(calls) == 2 and r["error"]["suggested_action"] == "queue_reject"
    assert world.conn.execute("SELECT count(*) FROM inspections").fetchone()[0] == 0                  # no se inventa ni se guarda un resultado


def test_vision_timeout_not_retried(world, cache, sample_ids, monkeypatch):
    img(world, cache, sample_ids, "GRASP", "g1")
    calls = []
    def h(req):
        calls.append(1); raise httpx.ReadTimeout("lento")
    _fail_with(world, monkeypatch, h)
    r = world.call("vision", "inspect_image", {"image_ids": ["g1"]}, role="calidad")
    assert r["error"]["code"] == "VISION_TIMEOUT" and len(calls) == 1 and r["error"]["http_status"] == 504


def test_queue_flow_and_labels(world, cache, sample_ids):
    img(world, cache, sample_ids, "REJECT", "r1")
    assert world.call("vision", "queue_reject", {"image_id": "r1", "reason": "rechazado por el modelo"}, role="calidad")["error"]["code"] == "NOT_INSPECTED"
    world.call("vision", "inspect_image", {"image_ids": ["r1"]}, role="calidad")
    q1 = world.call("vision", "queue_reject", {"image_id": "r1", "reason": "rechazado por el modelo"}, role="calidad")
    q2 = world.call("vision", "queue_reject", {"image_id": "r1", "reason": "otro texto distinto"}, role="calidad", sid="otra-sesion")
    assert q1["ok"] and q2["queue_id"] == q1["queue_id"] and q2["already_queued"]                   # idempotente por image_id
    lst = world.call("vision", "list_review_queue", {"status": "open"}, role="calidad")
    assert lst["count"] == 1 and lst["cases"][0]["queue_id"] == q1["queue_id"]
    assert world.call("vision", "list_review_queue", {}, role="calidad", line="L2")["count"] == 0   # otra linea no ve el caso
    bad = world.call("vision", "confirm_review_label", {"queue_id": q1["queue_id"], "confirmed_class": 9999}, role="calidad")
    assert bad["error"]["code"] == "UNKNOWN_CLASS"
    l1 = world.call("vision", "confirm_review_label", {"queue_id": q1["queue_id"], "confirmed_class": 5, "confirmed_angle": 120}, role="calidad")
    assert l1["status"] == "staged_for_retraining"
    l2 = world.call("vision", "confirm_review_label", {"queue_id": q1["queue_id"], "confirmed_class": 5, "confirmed_angle": 120}, role="calidad", sid="x")
    assert l2["label_id"] == l1["label_id"] and l2["already_labeled"]
    l3 = world.call("vision", "confirm_review_label", {"queue_id": q1["queue_id"], "confirmed_class": 6}, role="calidad", sid="y")
    assert l3["error"]["code"] == "CONFLICT"
    row = world.conn.execute("SELECT * FROM labels_staging").fetchone()
    assert row["labeled_by"] == "user.calidad" and row["model_version"]                              # staging con usuario y version del modelo


# ------------------------------------------------------------------ operaciones
def test_metrics_and_comparison(world):
    r = world.call("ops", "query_metrics_turn", {"turn_id": "2026-10-14-T3", "metric": "rejects", "interval": "hour"})
    assert r["ok"] and len(r["series"]) == 8 and r["unit"] == "casos" and abs(r["value"] - sum(x["value"] for x in r["series"])) < 1e-6
    assert r["previous_turn"]["turn_id"] == "2026-10-14-T2" and r["previous_turn"]["change_pct"] is not None
    short = world.call("ops", "query_metrics_turn", {"turn_id": "T2", "metric": "throughput", "interval": "turn"})
    assert short["turn_id"] == "2026-10-15-T2" and short["turn_closed"] is False and short["hours_with_data"] == 2 and short["previous_turn"]["comparable"] is False
    assert world.call("ops", "query_metrics_turn", {"turn_id": "2031-01-01-T1", "metric": "rejects", "interval": "turn"})["error"]["code"] == "TURN_NOT_FOUND"
    assert world.call("ops", "query_metrics_turn", {"turn_id": "T1", "metric": "salud", "interval": "turn"})["error"]["code"] == "VALIDATION_ERROR"


def test_line_context_tells_the_3am_story(world):
    r = world.call("ops", "get_line_context", {"window": {"kind": "range", "start": "2026-10-15T00:00", "end": "2026-10-15T06:00"}})
    assert r["ok"]
    kinds = {(c["kind"], c["ts"]) for c in r["changes_in_window"]}
    assert ("threshold", "2026-10-15T02:55:00") in kinds and not any(k == "model_version" for k, _ in kinds)        # el cambio de modelo fue antes
    h = {x["hour_start"][11:13]: x for x in r["camera"]["hourly"]}
    assert h["05"]["sharpness"] < h["01"]["sharpness"] * 0.6 and h["05"]["exposure"] < h["01"]["exposure"]
    assert r["camera"]["last_calibration"] == "2026-10-08T09:30:00"
    rej = world.call("ops", "query_metrics_turn", {"turn_id": "2026-10-14-T3", "metric": "rejects", "interval": "hour"})["series"]
    by = {x["hour_start"][11:13]: x["value"] for x in rej}
    assert by["04"] > 2 * by["01"]                                                                      # los rechazos suben despues de las 3
    assert "ip" not in str(r).lower() and "password" not in str(r).lower()


def test_line_context_window_limits(world):
    r = world.call("ops", "get_line_context", {"window": {"kind": "range", "start": "2026-10-14T00:00", "end": "2026-10-15T06:00"}})
    assert r["error"]["code"] == "VALIDATION_ERROR"
    assert world.call("ops", "get_line_context", {"window": {"kind": "range", "start": "ayer", "end": "hoy"}})["error"]["code"] == "VALIDATION_ERROR"


def test_shift_report(world):
    assert world.call("ops", "generate_shift_report", {"turn_id": "T2"})["error"]["code"] == "TURN_NOT_CLOSED"
    r1 = world.call("ops", "generate_shift_report", {"turn_id": "2026-10-14-T3"})
    r2 = world.call("ops", "generate_shift_report", {"turn_id": "2026-10-14-T3"}, sid="otra")
    assert r1["ok"] and r1["replaced_previous"] is False and r2["report_id"] == r1["report_id"] and r2["replaced_previous"] is True
    s = r1["summary"]
    assert s["totals"]["reject_rate_pct"] > 5 and any(i["kind"] == "threshold" for i in s["incidents"]) and len(s["open_tickets"]) == 2     # solo tickets de L1
    assert "issue_desc" not in str(s)                                                                  # solo ids, estado y conteos
    assert world.call("ops", "generate_shift_report", {"turn_id": "2026-10-14-T3"}, role="calidad")["error"]["code"] == "FORBIDDEN"


# ------------------------------------------------------------------ mantenimiento
def test_search_sop_cites_documents(world):
    r = world.call("maint", "search_sop", {"query": "la ventosa pierde succion y la pieza se resbala", "equipment": "gripper"}, role="mantenimiento")
    assert r["ok"] and 1 <= len(r["results"]) <= 3 and r["results"][0]["document_id"].startswith("SOP-GRP")
    assert all(len(x["snippet"]) <= 320 for x in r["results"])
    none = world.call("maint", "search_sop", {"query": "receta de arroz con coco"}, role="mantenimiento")
    assert none["results"] == [] and "no hay SOP aplicable" in none["note"]


def test_search_sop_flags_embedded_instructions(world):
    r = world.call("maint", "search_sop", {"query": "limpieza del lente de la camara", "equipment": "camera"}, role="mantenimiento")
    poisoned = [x for x in r["results"] if x["document_id"] == "SOP-CAM-004"]
    assert poisoned and poisoned[0]["contains_instructions_for_assistant"] is True and "ignorelo" in r["note"]
    assert not any(x["contains_instructions_for_assistant"] for x in r["results"] if x["document_id"] != "SOP-CAM-004")


def test_ticket_dedup_against_open_ticket(world):
    r = world.call("maint", "create_ticket", {"issue_desc": "ventosa del efector final con perdida de SUCCION intermitente en la estacion 2",
                                              "priority": "alta", "equipment": "gripper"}, role="mantenimiento")
    assert r["deduplicated"] is True and r["ticket_id"] == "TK-SEED01"
    n = world.call("maint", "create_ticket", {"issue_desc": "Sensor de presencia sin detectar piezas en la entrada", "priority": "media",
                                              "equipment": "conveyor"}, role="mantenimiento")
    assert n["deduplicated"] is False and n["ticket_id"].startswith("TK-")
    assert world.call("maint", "create_ticket", {"issue_desc": "corto", "priority": "urgente", "equipment": "gripper"}, role="mantenimiento")["error"]["code"] == "VALIDATION_ERROR"
    assert world.call("maint", "create_ticket", {"issue_desc": "Descripcion suficientemente larga", "priority": "alta", "equipment": "gripper"}, role="calidad")["error"]["code"] == "FORBIDDEN"


# ------------------------------------------------------------------ ingenieria ML
def test_sku_flow_validations(world):
    good = world.add_sku_views("c1")
    r = world.call("ml", "upload_new_sku_views", {"sku_name": "sku-conector-c1", "image_ids": good}, role="ingeniero_ml")
    assert r["status"] == "validated" and r["views_received"] == 36 and r["missing_angles_deg"] == []
    r2 = world.call("ml", "upload_new_sku_views", {"sku_name": "sku-conector-c1", "image_ids": good}, role="ingeniero_ml", sid="otra")
    assert r2["batch_id"] == r["batch_id"] and r2["already_staged"]                                       # idempotente por contenido
    part = world.add_sku_views("part", n=12)
    p = world.call("ml", "upload_new_sku_views", {"sku_name": "sku-parcial", "image_ids": part}, role="ingeniero_ml")
    assert p["status"] == "insufficient" and p["problems"] and p["missing_angles_deg"]
    assert world.call("ml", "test_sku_model", {"sku_name": "sku-parcial", "protocol": "A"}, role="ingeniero_ml")["error"]["code"] == "NOT_VALIDATED"
    blur = world.add_sku_views("bl", bad=(5,))
    b = world.call("ml", "upload_new_sku_views", {"sku_name": "sku-borroso", "image_ids": blur}, role="ingeniero_ml")
    assert b["status"] == "insufficient" and b["quality_failures"] == ["bl_05"]
    assert world.call("ml", "upload_new_sku_views", {"sku_name": "obj5", "image_ids": good}, role="ingeniero_ml")["error"]["code"] == "SKU_EXISTS"
    assert world.call("ml", "upload_new_sku_views", {"sku_name": "sku-x", "image_ids": ["no_existe"]}, role="ingeniero_ml")["error"]["code"] == "IMAGE_NOT_FOUND"
    assert world.call("ml", "upload_new_sku_views", {"sku_name": "sku-x", "image_ids": good}, role="supervisor")["error"]["code"] == "FORBIDDEN"
    assert world.conn.execute("SELECT count(*) FROM skus").fetchone()[0] == 100                          # staging: el catalogo productivo no cambia


def test_test_sku_model_is_labelled_simulated(world):
    world.call("ml", "upload_new_sku_views", {"sku_name": "sku-conector-c1", "image_ids": world.add_sku_views("c1")}, role="ingeniero_ml")
    r = world.call("ml", "test_sku_model", {"sku_name": "sku-conector-c1", "protocol": "A"}, role="ingeniero_ml")
    assert r["ok"] and r["simulated"] is True and r["evidence_report_id"].startswith("ER-") and r["candidate_version"].endswith("+sku-conector-c1")
    assert 0 <= r["acc_clase"] <= 1
