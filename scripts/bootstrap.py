"""Prepara los datos de la demo SI FALTAN (idempotente). Lo usan `docker compose up` (servicio bootstrap) y scripts/run_stack.py.

  python scripts/bootstrap.py            # siembra solo si no existe la base o le faltan las imagenes de demostracion
  python scripts/bootstrap.py --force    # vuelve a sembrar desde cero (borra la base operativa)

Con data/cache/coil100.npz se siembran base + imagenes de demostracion + cola de revision (scripts/seed_demo.py). Sin ese cache
(se genera con `python src/prepare_data.py`, ver README) se siembra solo la base: el asistente funciona y las imagenes se cargan
desde la pantalla, pero los identificadores img_grasp_01, img_reject_01, ... de los escenarios de la demo no existiran.
"""
import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from services.common import db, fixtures                                   # noqa: E402
from services.common.config import Config                                  # noqa: E402
from services.common.envfile import load_env                               # noqa: E402


def seeded(cfg):
    if not cfg.db_path.exists():
        return False
    try:
        c = sqlite3.connect(str(cfg.db_path))
        n = c.execute("SELECT count(*) FROM images WHERE image_id='img_grasp_01'").fetchone()[0]
        c.close()
        return n == 1 and (cfg.image_store / "img_grasp_01.png").exists()
    except sqlite3.Error:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    load_env()
    cfg = Config.load()
    if not a.force and seeded(cfg):
        print("bootstrap: datos de demostracion ya presentes; nada que hacer")
        return 0
    cache = ROOT / "data" / "cache" / "coil100.npz"
    if cache.exists():
        import seed_demo
        sys.argv = ["seed_demo.py", "--reset"]
        seed_demo.main()
        return 0
    if cfg.db_path.exists() and not a.force:
        print("bootstrap: hay base operativa pero falta data/cache/coil100.npz; no se tocan los datos existentes")
        return 0
    for ext in ("", "-wal", "-shm"):
        Path(str(cfg.db_path) + ext).unlink(missing_ok=True)
    conn = db.init_schema(db.connect(cfg.db_path))
    fixtures.seed_db(conn, cfg, ROOT / "results")
    print("bootstrap: base sembrada SIN imagenes de demostracion (falta data/cache/coil100.npz: python src/prepare_data.py). "
          "Carga imagenes desde la pantalla; los escenarios con img_* de ejemplo no funcionaran.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
