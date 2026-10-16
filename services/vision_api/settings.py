"""Configuracion del servicio de vision: todo viene del entorno (ver .env.example); nada de valores incrustados."""
import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env(name, default):
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    model_dir: Path
    src_dir: Path
    max_image_bytes: int
    max_images_per_request: int
    # umbrales de calidad (heuristicas; ver services/vision_api/quality.py)
    q_blur_min: float
    q_dark_max_mean: float
    q_bright_min_mean: float
    q_contrast_min_std: float
    q_min_side: int
    # solo para demostrar escenarios adversos (error 5xx / timeout); apagado por defecto
    fault_injection: bool
    fault_mode: str

    @classmethod
    def load(cls):
        root = Path(_env("VISION_REPO_ROOT", str(REPO_ROOT)))
        return cls(
            model_dir=Path(_env("VISION_MODEL_DIR", str(root / "models" / "protocolo_A"))),
            src_dir=Path(_env("VISION_SRC_DIR", str(root / "src"))),
            max_image_bytes=int(_env("VISION_MAX_IMAGE_BYTES", str(5 * 1024 * 1024))),
            max_images_per_request=int(_env("VISION_MAX_IMAGES_PER_REQUEST", "5")),
            q_blur_min=float(_env("VISION_Q_BLUR_MIN", "25")),
            q_dark_max_mean=float(_env("VISION_Q_DARK_MAX_MEAN", "15")),
            q_bright_min_mean=float(_env("VISION_Q_BRIGHT_MIN_MEAN", "180")),
            q_contrast_min_std=float(_env("VISION_Q_CONTRAST_MIN_STD", "10")),
            q_min_side=int(_env("VISION_Q_MIN_SIDE", "32")),
            fault_injection=_env("VISION_FAULT_INJECTION", "0") == "1",
            fault_mode=_env("VISION_FAULT_MODE", ""),         # "500" o "delay:<s>"; solo si fault_injection=1
        )
