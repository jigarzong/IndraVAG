"""API HTTP del servicio de vision (independiente del agente). Reto 4: servicio de inferencia con API propia."""
import asyncio
from contextlib import asynccontextmanager
import time
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .engine import ServiceEngine
from .quality import assess
from .settings import Settings

def create_app(settings=None):
    cfg = settings or Settings.load()
    state = {}

    @asynccontextmanager
    async def lifespan(_):
        state["engine"] = ServiceEngine(cfg)      # el modelo se carga al arrancar, no en la primera solicitud
        yield

    app = FastAPI(title="Indra Vision Service", version="1.0.0", lifespan=lifespan)

    def eng():
        if "engine" not in state:
            state["engine"] = ServiceEngine(cfg)
        return state["engine"]

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": "engine" in state}

    @app.get("/model")
    def model():
        return eng().info()

    @app.post("/v1/predict")
    async def predict(files: list[UploadFile] = File(...), x_fault: str | None = Header(default=None)):
        rid = uuid.uuid4().hex[:12]
        fault = (x_fault or cfg.fault_mode) if cfg.fault_injection else ""   # solo para la demo de escenarios adversos
        if fault == "500":
            raise HTTPException(500, "fallo inyectado")
        if fault.startswith("delay:"):
            await asyncio.sleep(float(fault.split(":")[1]))
        if not files:
            raise HTTPException(422, "se requiere al menos una imagen")
        if len(files) > cfg.max_images_per_request:
            raise HTTPException(413, f"maximo {cfg.max_images_per_request} imagenes por solicitud")
        t0 = time.time()
        items, batch, pos = [], [], []
        for f in files:
            raw = await f.read()
            it = dict(filename=f.filename, prediction=None, evidence=[], quality=None, error=None)
            if len(raw) > cfg.max_image_bytes:
                raise HTTPException(413, f"{f.filename}: supera {cfg.max_image_bytes} bytes")
            bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) if raw else None
            if bgr is None:
                it["error"] = dict(code="INVALID_IMAGE", message="no se pudo decodificar como imagen")
            else:
                side = min(bgr.shape[:2])
                rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (128, 128), interpolation=cv2.INTER_AREA)
                it["quality"] = assess(rgb, side, cfg)
                if it["quality"]["ok"]:
                    batch.append(rgb); pos.append(len(items))
                else:                                                # imagen de baja calidad: no se llama al modelo
                    it["recommended_action"] = "RECAPTURE"
            items.append(it)
        if batch:
            res, ev = eng().predict(np.stack(batch))
            for j, k in enumerate(pos):
                r = res[j]
                items[k]["prediction"], items[k]["evidence"] = r, ev[j]
        return {"request_id": rid, "thresholds_version": eng().version, "latency_ms": round((time.time() - t0) * 1000, 1), "results": items}

    return app


app = create_app()
