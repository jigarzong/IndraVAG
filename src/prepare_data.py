"""Construye el cache de COIL-100 (imagenes, etiquetas, huella SHA-256) y los descriptores.
Uso (desde la raiz del repo):  INDRA_DATA=data/COIL-100 python src/prepare_data.py
"""
import os, re, sys, hashlib, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import hog_features, color_features

root = os.environ.get("INDRA_DATA", "data/COIL-100")
files = sorted((f for f in os.listdir(root) if f.lower().endswith(".png")),
               key=lambda f: tuple(int(x) for x in re.findall(r"\d+", f)[:2]))
assert len(files) == 7200, f"se esperaban 7200 imagenes, hay {len(files)}"
h = hashlib.sha256(); imgs, obj, ang = [], [], []
for f in files:
    raw = open(os.path.join(root, f), "rb").read(); h.update(raw)
    imgs.append(cv2.cvtColor(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    o, a = (int(x) for x in re.findall(r"\d+", f)[:2]); obj.append(o); ang.append(a)
imgs, obj, ang = np.stack(imgs), np.array(obj), np.array(ang)
os.makedirs("data/cache", exist_ok=True)
np.savez_compressed("data/cache/coil100.npz", imgs=imgs, obj=obj, ang=ang, names=np.array(files), sha=h.hexdigest())
np.savez_compressed("data/cache/features.npz", hog=hog_features(imgs), color=color_features(imgs))
pix = np.stack([cv2.resize(i, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255 for i in imgs])
np.savez_compressed("data/cache/features_pix.npz", pix=pix)
print("SHA-256 del dataset:", h.hexdigest())
