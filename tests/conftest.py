import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def cache():
    p = ROOT / "data" / "cache" / "coil100.npz"
    if not p.exists():
        pytest.skip("falta data/cache/coil100.npz (python src/prepare_data.py)")
    return np.load(p)


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    """Cada prueba usa su propia base y su almacen de imagenes; los secretos son de prueba."""
    monkeypatch.setenv("JWT_SECRET", "test-jwt-secret-0123456789-abcdefghijklmnop")
    monkeypatch.setenv("APPROVAL_SECRET", "test-approval-secret-0123456789-abcdefghijk")
    monkeypatch.setenv("OPS_DB_PATH", str(tmp_path / "ops.db"))
    monkeypatch.setenv("IMAGE_STORE_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("VISION_RETRY_WAIT_SECONDS", "0")


@pytest.fixture(scope="session")
def vision_client():
    from fastapi.testclient import TestClient
    from services.vision_api.app import create_app
    from services.vision_api.settings import Settings
    with TestClient(create_app(Settings.load())) as c:
        yield c


@pytest.fixture(scope="session")
def sample_ids(cache, vision_client):
    """Indices de COIL-100 con decision GRASP y REJECT segun el propio servicio (se calculan una vez)."""
    idx = np.load(ROOT / "results" / "A_out.npz")["test_idx"][:80]
    out = {"GRASP": None, "REJECT": None}
    for i in idx:
        ok, buf = cv2.imencode(".png", cv2.cvtColor(cache["imgs"][i], cv2.COLOR_RGB2BGR))
        d = vision_client.post("/v1/predict", files=[("files", ("x.png", buf.tobytes(), "image/png"))]).json()["results"][0]["prediction"]["decision"]
        if d in out and out[d] is None:
            out[d] = int(i)
        if all(out.values()):
            break
    return out


class World:
    def __init__(self, cfg, conn, servers):
        self.cfg, self.conn, self.s = cfg, conn, servers

    def token(self, role, user=None, line="L1", sid="sess1"):
        from services.common import auth
        return auth.issue_token(self.cfg.jwt_secret, user or f"user.{role}", role, line, sid)

    def call(self, server, tool, args, role="supervisor", user=None, line="L1", sid="sess1"):
        return self.s[server].invoke(tool, args, self.token(role, user, line, sid))

    def add_image(self, image_id, rgb, line="L1", angle=None):
        self.cfg.image_store.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        (self.cfg.image_store / f"{image_id}.png").write_bytes(buf.tobytes())
        from services.vision_api.quality import assess
        from services.vision_api.settings import Settings
        q = assess(cv2.resize(rgb, (128, 128)), min(rgb.shape[:2]), Settings.load())
        self.conn.execute("INSERT OR REPLACE INTO images VALUES(?,?,?,?,?,?,?,?,?)", (image_id, line, f"{image_id}.png", hashlib.sha256(buf.tobytes()).hexdigest(),
                                                                                    "test", "2026-10-15T10:00:00+00:00", angle, int(q["ok"]), ",".join(q["flags"])))

    def add_sku_views(self, prefix, n=36, bad=(), step=10.0):
        ids = []
        for k in range(n):
            iid = f"{prefix}_{k:02d}"
            self.conn.execute("INSERT OR REPLACE INTO images VALUES(?,?,?,?,?,?,?,?,?)",
                              (iid, "L1", f"{iid}.png", hashlib.sha256(iid.encode()).hexdigest(), "test", "2026-10-15T10:00:00+00:00", k * step,
                               0 if k in bad else 1, "BLURRY" if k in bad else ""))
            ids.append(iid)
        return ids


@pytest.fixture
def world(env, vision_client):
    from services.common import db, fixtures
    from services.common.config import Config
    from services.mcp_servers import maintenance_server, ml_eng_server, ops_server, vision_server
    cfg = Config.load()
    conn = db.init_schema(db.connect(cfg.db_path))
    fixtures.seed_db(conn, cfg, ROOT / "results")
    vision_server.CLIENT_FACTORY = lambda: vision_client
    servers = {"vision": vision_server.make_server(cfg=cfg, conn=conn), "ops": ops_server.make_server(cfg=cfg, conn=conn),
               "maint": maintenance_server.make_server(cfg=cfg, conn=conn), "ml": ml_eng_server.make_server(cfg=cfg, conn=conn)}
    yield World(cfg, conn, servers)
    vision_server.CLIENT_FACTORY = None
