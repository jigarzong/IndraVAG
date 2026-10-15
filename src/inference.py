"""Funcion de inferencia del motor de vision (Reto 2): imagen RGB -> decision para el robot."""
import sys, os
import json, joblib, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import hog_features, color_features, circ_err, angle_posterior, l2n
from experiment import raw_embed


class VisionEngine:
    def __init__(self, path="models/protocolo_A"):
        b = joblib.load(f"{path}/bundle.joblib")
        self.__dict__.update(b)

    def features(self, imgs):
        return np.hstack([hog_features(imgs), color_features(imgs)])

    def predict(self, imgs):
        """imgs: arreglo (n,128,128,3) uint8 RGB. Devuelve lista de dicts con el contrato de salida."""
        X = self.features(imgs)
        pr = self.pipe.predict_proba(X)
        cls = self.pipe.classes_[pr.argmax(1)]
        conf = pr.max(1)
        est, unc, dmin = angle_posterior(raw_embed(X), self.Z_train.astype(np.float32), self.ang_train, self.obj_train, cls, self.tau, self.k)
        t = self.thr
        out = []
        for i in range(len(X)):
            class_ok = conf[i] >= t["confianza_clase_cruda"] and dmin[i] <= t["distancia_vista_max"]
            angle_ok = unc[i] <= t["incertidumbre_angulo_deg"]
            dec = "GRASP" if (class_ok and angle_ok) else ("RETRY_VIEW" if class_ok else "REJECT")
            out.append(dict(object_id=int(cls[i]), confidence=float(self.iso.predict([conf[i]])[0]), angle_deg=float(est[i]),
                            angle_uncertainty_deg=float(unc[i]), nearest_view_distance=float(dmin[i]), decision=dec,
                            thresholds_version=self.version))
        return out
