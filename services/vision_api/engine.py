"""Envoltura del VisionEngine del Reto 2 (src/inference.py). No reimplementa el modelo: lo carga y lo invoca."""
import json
import sys
from pathlib import Path
import numpy as np


class ServiceEngine:
    def __init__(self, cfg):
        sys.path.insert(0, str(cfg.src_dir))
        from inference import VisionEngine          # noqa: E402  (src/ viene de la configuracion)
        from experiment import raw_embed            # noqa: E402
        self._raw_embed = raw_embed
        self.engine = VisionEngine(str(cfg.model_dir))
        self.model_dir = str(cfg.model_dir)
        self.meta = json.loads((Path(cfg.model_dir) / "thresholds.json").read_text())   # huella del dataset, protocolo, descriptores

    @property
    def version(self):
        return self.engine.version

    def info(self):
        e = self.engine
        return dict(thresholds_version=e.version, n_classes=int(len(e.pipe.classes_)), n_train_views=int(len(e.ang_train)),
                    thresholds={k: e.thr[k] for k in ("confianza_clase_cruda", "incertidumbre_angulo_deg", "distancia_vista_max")},
                    dataset_sha256=self.meta.get("dataset_sha256"), protocol=self.meta.get("protocolo"), descriptors=self.meta.get("descriptores"))

    def predict(self, imgs, n_evidence=3):
        """imgs: (n,128,128,3) uint8 RGB. Devuelve (resultados, evidencia): vistas de entrenamiento mas parecidas de la clase predicha."""
        res = self.engine.predict(imgs)
        e = self.engine
        X = e.features(imgs)
        score = e.pipe.predict_proba(X).max(1)                # puntaje crudo de clase (antes de calibrar), para explicar el rechazo
        Z = self._raw_embed(X)
        ev = []
        for i, r in enumerate(res):
            r["class_score"] = float(score[i])
            t = e.thr
            if r["decision"] == "REJECT":
                r["decision_reason"] = "LOW_CLASS_SCORE" if score[i] < t["confianza_clase_cruda"] else "FAR_FROM_KNOWN_VIEWS"
            elif r["decision"] == "RETRY_VIEW":
                r["decision_reason"] = "ANGLE_UNCERTAIN"
            else:
                r["decision_reason"] = "THRESHOLDS_OK"
            ti = np.where(e.obj_train == r["object_id"])[0]
            s = Z[i] @ e.Z_train[ti].astype(np.float32).T
            top = np.argsort(-s)[:n_evidence]
            ev.append([dict(object_id=r["object_id"], angle_deg=float(e.ang_train[ti][j]), cosine_similarity=float(s[j])) for j in top])
        return res, ev
