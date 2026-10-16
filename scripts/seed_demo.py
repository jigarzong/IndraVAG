"""Siembra el MVP con datos SIMULADOS reproducibles: base operativa, imagenes de demostracion y casos en la cola de revision.

  python scripts/seed_demo.py --reset

Necesita data/cache/coil100.npz (python src/prepare_data.py) y el modelo exportado. Las imagenes salen de COIL-100 (o de skimage.data para
los casos fuera de dominio) y se escriben en IMAGE_STORE_DIR (no se versionan).
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np

from services.common import audit, auth, db, fixtures
from services.common.config import Config
from services.mcp_servers import vision_server as vs
from services.vision_api.app import create_app
from services.vision_api.quality import assess
from services.vision_api.settings import Settings


def write_img(cfg, conn, image_id, rgb, line, angle=None, by="seed"):
    cfg.image_store.mkdir(parents=True, exist_ok=True)
    name = f"{image_id}.png"
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    (cfg.image_store / name).write_bytes(buf.tobytes())
    q = assess(cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA), min(rgb.shape[:2]), Settings.load())
    conn.execute("INSERT OR REPLACE INTO images VALUES(?,?,?,?,?,?,?,?,?)",
                 (image_id, line, name, hashlib.sha256(buf.tobytes()).hexdigest(), by, audit.now_iso(), angle, int(q["ok"]), ",".join(q["flags"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()
    cfg = Config.load()
    if a.reset and cfg.db_path.exists():
        for ext in ("", "-wal", "-shm"):
            Path(str(cfg.db_path) + ext).unlink(missing_ok=True)
    conn = db.init_schema(db.connect(cfg.db_path))
    fixtures.seed_db(conn, cfg, ROOT / "results")

    cache = np.load(ROOT / "data" / "cache" / "coil100.npz")
    imgs, obj, ang = cache["imgs"], cache["obj"], cache["ang"]
    from skimage import data as skdata

    from fastapi.testclient import TestClient
    client = TestClient(create_app(Settings.load()))
    client.__enter__()
    # clasificar candidatos con el propio servicio para elegir ejemplos de cada decision
    rng = np.random.default_rng(42)
    test_idx = np.load(ROOT / "results" / "A_out.npz")["test_idx"]
    cand = rng.choice(test_idx, 400, replace=False)
    picks = {"GRASP": [], "RETRY_VIEW": [], "REJECT": []}
    for i in cand:
        ok, buf = cv2.imencode(".png", cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR))
        r = client.post("/v1/predict", files=[("files", ("x.png", buf.tobytes(), "image/png"))]).json()["results"][0]["prediction"]
        picks[r["decision"]].append(int(i))
    print({k: len(v) for k, v in picks.items()})
    for k, nm, n in (("GRASP", "grasp", 6), ("RETRY_VIEW", "retry", 2), ("REJECT", "reject", 6)):
        for j, i in enumerate(picks[k][:n], 1):
            write_img(cfg, conn, f"img_{nm}_{j:02d}", imgs[i], "L1")
    # una imagen buena para la otra linea (aislamiento)
    write_img(cfg, conn, "img_l2_grasp_01", imgs[picks["GRASP"][-1]], "L2")
    base = imgs[picks["GRASP"][0]]
    write_img(cfg, conn, "img_blur_01", cv2.GaussianBlur(base, (0, 0), 6), "L1")
    write_img(cfg, conn, "img_dark_01", (base * 0.04).astype(np.uint8), "L1")
    write_img(cfg, conn, "img_blank_01", np.zeros((128, 128, 3), np.uint8), "L1")
    for j, nm in enumerate(("astronaut", "coffee", "chelsea", "rocket"), 1):
        im = getattr(skdata, nm)()
        s = min(im.shape[:2]); y, x = (im.shape[0] - s) // 2, (im.shape[1] - s) // 2
        write_img(cfg, conn, f"img_ood_{j:02d}", cv2.resize(im[y:y + s, x:x + s], (128, 128), interpolation=cv2.INTER_AREA), "L1")
    write_img(cfg, conn, "img_corrupt_01", np.zeros((8, 8, 3), np.uint8), "L1")                    # demasiado pequena
    (cfg.image_store / "img_notimage_01.png").write_bytes(b"esto no es una imagen")                # archivo no decodificable
    conn.execute("INSERT OR REPLACE INTO images VALUES('img_notimage_01','L1','img_notimage_01.png',?,?,?,NULL,1,'')",
                 (hashlib.sha256(b"x").hexdigest(), "seed", audit.now_iso()))
    # lotes de SKU nuevo: 36 vistas cada 10 grados de un objeto de COIL (SIMULADO: el objeto ya esta en el catalogo del modelo)
    def sku_views(prefix, o, n=36, blur_idx=None):
        ids = []
        for k in range(n):
            sel = np.where((obj == o) & (ang == (k * 10) % 360))[0][0]
            im = imgs[sel]
            if blur_idx is not None and k == blur_idx:
                im = cv2.GaussianBlur(im, (0, 0), 6)
            iid = f"{prefix}_{k:02d}"
            write_img(cfg, conn, iid, im, "L1", angle=float(k * 10), by="ing.vega")
            ids.append(iid)
        return ids
    sku_views("skuc1", 17)
    sku_views("skug2", 23)
    sku_views("skupart", 31, n=12)
    sku_views("skublur", 41, blur_idx=5)
    # cola de revision sembrada con el pipeline real de las tools (inspect_image -> queue_reject)
    vs.CLIENT_FACTORY = lambda: client
    tok = auth.issue_token(cfg.jwt_secret, "cal.mora", "calidad", "L1", "seed-session")
    srv = vs.make_server(cfg=cfg, conn=conn)
    for iid in ("img_reject_01", "img_reject_02", "img_reject_03", "img_ood_01", "img_retry_01"):
        r = srv.invoke("inspect_image", {"image_ids": [iid]}, tok)
        assert r["ok"], r
        q = srv.invoke("queue_reject", {"image_id": iid, "reason": "rechazo del modelo; requiere revision humana"}, tok)
        assert q["ok"], q
    print("imagenes:", conn.execute("SELECT count(*) FROM images").fetchone()[0], "| cola:", conn.execute("SELECT count(*) FROM review_queue").fetchone()[0])
    print("auditoria:", audit.verify_chain(conn))
    client.__exit__(None, None, None)


if __name__ == "__main__":
    main()
