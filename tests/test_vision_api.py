"""Pruebas del servicio de vision: equivalencia con VisionEngine (Reto 2) y manejo de entradas invalidas."""
import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from services.vision_api.app import create_app
from services.vision_api.settings import Settings


def png(rgb):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    return buf.tobytes()


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(Settings.load())) as c:
        yield c


def post(client, blobs, **kw):
    files = [("files", (f"i{k}.png", b, "image/png")) for k, b in enumerate(blobs)]
    return client.post("/v1/predict", files=files, **kw)


def test_health_and_model(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    m = client.get("/model").json()
    assert m["thresholds_version"].startswith("A-") and m["n_classes"] == 100 and m["n_train_views"] == 3600
    assert m["protocol"] == "A" and m["dataset_sha256"].startswith("26635cc67c13")


def test_equivalence_with_engine(client, cache):
    """La API devuelve lo mismo que VisionEngine.predict llamado directamente (PNG sin perdida)."""
    from inference import VisionEngine
    from services.vision_api.settings import Settings
    eng = VisionEngine(str(Settings.load().model_dir))
    imgs = cache["imgs"][[3, 500, 2000, 4100, 7000]]
    direct = eng.predict(imgs)
    r = post(client, [png(i) for i in imgs])
    assert r.status_code == 200
    for d, it in zip(direct, r.json()["results"]):
        p = it["prediction"]
        assert p["object_id"] == d["object_id"]
        assert abs(p["confidence"] - d["confidence"]) < 1e-6
        assert abs(((p["angle_deg"] - d["angle_deg"] + 180) % 360) - 180) < 1e-3
        assert it["quality"]["ok"] and len(it["evidence"]) == 3
        assert p["thresholds_version"] == d["thresholds_version"]
        assert p["decision"] in {"GRASP", "RETRY_VIEW", "REJECT"} and p["decision"] == d["decision"]


def test_contract_fields(client, cache):
    it = post(client, [png(cache["imgs"][10])]).json()["results"][0]
    assert set(it["prediction"]) == {"object_id", "confidence", "angle_deg", "angle_uncertainty_deg", "nearest_view_distance",
                                     "decision", "thresholds_version", "decision_reason", "class_score"}


def test_undecodable_image_is_item_error(client):
    r = post(client, [b"esto no es una imagen"])
    assert r.status_code == 200
    it = r.json()["results"][0]
    assert it["error"]["code"] == "INVALID_IMAGE" and it["prediction"] is None


def test_blank_image_model_not_called(client):
    it = post(client, [png(np.zeros((128, 128, 3), np.uint8))]).json()["results"][0]
    assert "TOO_DARK" in it["quality"]["flags"] and it["prediction"] is None and it["recommended_action"] == "RECAPTURE"


def test_blurry_image_flagged(client, cache):
    blur = cv2.GaussianBlur(cache["imgs"][10], (0, 0), 6)
    it = post(client, [png(blur)]).json()["results"][0]
    assert "BLURRY" in it["quality"]["flags"] and it["prediction"] is None and it["recommended_action"] == "RECAPTURE"


def test_too_many_images(client, cache):
    assert post(client, [png(cache["imgs"][0])] * 6).status_code == 413


def test_no_files(client):
    assert client.post("/v1/predict").status_code == 422


def test_fault_injection_off_by_default(client, cache):
    r = post(client, [png(cache["imgs"][0])], headers={"X-Fault": "500"})
    assert r.status_code == 200          # sin VISION_FAULT_INJECTION=1 el encabezado se ignora
