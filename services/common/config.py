"""Configuracion compartida por los servidores MCP, el orquestador y la API de aprobaciones. Todo viene del entorno."""
import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _p(name, default):
    v = os.environ.get(name, default)
    p = Path(v)
    return p if p.is_absolute() else REPO_ROOT / p


@dataclass(frozen=True)
class Config:
    db_path: Path
    image_store: Path
    vision_url: str
    vision_timeout_s: float
    vision_retries_5xx: int
    vision_retry_wait_s: float
    jwt_secret: str
    approval_secret: str
    approval_ttl_min: int
    ticket_dedup_hours: int
    sku_min_views: int
    sku_max_angle_gap_deg: float
    promotion_min_acc_clase: float
    promotion_min_pct_within_10: float
    demo_today: str                     # fecha fija de la simulacion (el MVP no usa el reloj para que los casos sean reproducibles)

    @classmethod
    def load(cls):
        e = os.environ.get
        return cls(
            db_path=_p("OPS_DB_PATH", "data/ops.db"),
            image_store=_p("IMAGE_STORE_DIR", "data/uploads"),
            vision_url=e("VISION_SERVICE_URL", "http://localhost:8001"),
            vision_timeout_s=float(e("VISION_TIMEOUT_SECONDS", "5")),
            vision_retries_5xx=int(e("VISION_RETRIES_5XX", "1")),
            vision_retry_wait_s=float(e("VISION_RETRY_WAIT_SECONDS", "0.5")),
            jwt_secret=e("JWT_SECRET", ""),
            approval_secret=e("APPROVAL_SECRET", ""),
            approval_ttl_min=int(e("APPROVAL_TTL_MINUTES", "30")),
            ticket_dedup_hours=int(e("TICKET_DEDUP_HOURS", "24")),
            sku_min_views=int(e("SKU_MIN_VIEWS", "36")),
            sku_max_angle_gap_deg=float(e("SKU_MAX_ANGLE_GAP_DEG", "20")),
            promotion_min_acc_clase=float(e("PROMOTION_MIN_ACC_CLASE", "0.98")),
            promotion_min_pct_within_10=float(e("PROMOTION_MIN_PCT_WITHIN_10", "0.90")),
            demo_today=e("DEMO_TODAY", "2026-10-15"),
        )
