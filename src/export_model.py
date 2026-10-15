import sys, os
import json, joblib, os, numpy as np
from sklearn.isotonic import IsotonicRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment import *
d = np.load("data/cache/coil100.npz"); f = np.load("data/cache/features.npz")
obj, ang = d["obj"], d["ang"]
X = np.hstack([f["hog"], f["color"]])
P, _, _ = make_protocols(obj, ang)
tr, te = P["A"]
R = json.load(open("results/A.json")); o = np.load("results/A_out.npz")
pipe, grid = candidates()["SVM-RBF"]
params = {f"svc__{k}": v for k, v in R["comparacion_cv"][0]["mejores_params"].items()}
assert R["mejor_modelo"] == "SVM-RBF"
final = with_proba(pipe.set_params(**params)).fit(X[tr], obj[tr])
iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(o["oof_conf"], o["oof_ok"].astype(float))
os.makedirs("models/protocolo_A", exist_ok=True)
bundle = dict(pipe=final, iso=iso, Z_train=raw_embed(X[tr]).astype(np.float16), ang_train=ang[tr], obj_train=obj[tr],
              tau=R["angulo_params"]["tau"], k=R["angulo_params"]["k"], thr=R["umbrales"], version=f"A-{str(d['sha'])[:12]}")
joblib.dump(bundle, "models/protocolo_A/bundle.joblib", compress=3)
json.dump(dict(R["umbrales"], tau=bundle["tau"], k=bundle["k"], dataset_sha256=str(d["sha"]), protocolo="A", descriptores="hogcolor"),
          open("models/protocolo_A/thresholds.json", "w"), indent=1, default=lambda v: v.item() if hasattr(v, "item") else v)
print("ok", os.path.getsize("models/protocolo_A/bundle.joblib") / 1e6, "MB")
