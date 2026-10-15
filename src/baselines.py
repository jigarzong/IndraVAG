"""Reto 2 - referencias: (1) base simple 1-NN sobre pixeles, (2) CNN pequena (Keras, CPU).
Mismas particiones que los protocolos; la prueba solo se evalua una vez al final."""
import sys, json, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np, cv2
from lib import *
from experiment import view_folds, TOL

d = np.load("data/cache/coil100.npz")
imgs, obj, ang = d["imgs"], d["obj"], d["ang"]
P, arc, unk = make_protocols(obj, ang)


def pix(im, s=32):
    return cv2.resize(im, (s, s), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255.0


def nn_baseline(p):
    tr, te = P[p]
    Xp = np.stack([pix(i) for i in imgs])
    Xtr, Xte = Xp[tr], Xp[te]
    t0 = time.time()
    Xn = l2n(Xtr - Xtr.mean(1, keepdims=True)); Qn = l2n(Xte - Xte.mean(1, keepdims=True))
    t_fit = time.time() - t0
    t0 = time.time()
    S = Qn @ Xn.T
    j = S.argmax(1)
    t_inf = (time.time() - t0) / len(te) * 1000
    pred, est = obj[tr][j], ang[tr][j]
    R = dict(modelo="1-NN pixeles 32x32 (correlacion)", t_ajuste_s=round(t_fit, 2), ms_por_imagen=round(t_inf, 3))
    if p != "D":
        err = circ_err(est, ang[te]); ok = pred == obj[te]
        R.update(acc_clase=float(ok.mean()), mae_angulo=float(err.mean()), pct_dentro_10=float((err <= TOL).mean() * 100),
                 agarre_seguro=float((ok & (err <= TOL)).mean() * 100))
    else:
        known = []
        for a_, b_ in view_folds("D", ang[tr]):
            known.append((Xn[b_] @ Xn[a_].T).max(1))
        known = np.concatenate(known)
        R.update(auroc_similitud=auroc(known, S.max(1)))
    return R


def build_cnn(n_cls):
    import tensorflow as tf
    from tensorflow.keras import layers as L, Model
    inp = L.Input((64, 64, 3))
    x = inp
    for f in (32, 64, 128):
        x = L.Conv2D(f, 3, padding="same", activation="relu")(x); x = L.BatchNormalization()(x); x = L.MaxPool2D()(x)
    x = L.GlobalAveragePooling2D()(x); x = L.Dense(256, activation="relu")(x); x = L.Dropout(0.3)(x)
    c = L.Dense(n_cls, activation="softmax", name="cls")(x)
    a = L.Dense(2, activation="tanh", name="ang")(x)       # (sen, cos)
    m = Model(inp, [c, a])
    m.compile("adam", loss={"cls": "sparse_categorical_crossentropy", "ang": "mse"}, loss_weights={"cls": 1.0, "ang": 2.0})
    return m


def cnn_ref(p, epochs=int(os.environ.get("CNN_EPOCHS", 25))):
    import tensorflow as tf
    tf.random.set_seed(SEED)
    tr, te = P[p]
    Xp = np.stack([cv2.resize(i, (64, 64), interpolation=cv2.INTER_AREA) for i in imgs]).astype(np.float32) / 255.0
    cls_ids = np.unique(obj[tr]); cmap = {c: i for i, c in enumerate(cls_ids)}
    ytr = np.array([cmap[c] for c in obj[tr]])
    atr = np.deg2rad(ang[tr]); A = np.stack([np.sin(atr), np.cos(atr)], 1)
    m = build_cnn(len(cls_ids))
    t0 = time.time()
    m.fit(Xp[tr], {"cls": ytr, "ang": A}, epochs=epochs, batch_size=64, verbose=0)
    t_fit = time.time() - t0
    t0 = time.time()
    pc, pa = m.predict(Xp[te], batch_size=256, verbose=0)
    t_inf = (time.time() - t0) / len(te) * 1000
    est = np.rad2deg(np.arctan2(pa[:, 0], pa[:, 1])) % 360
    pred = cls_ids[pc.argmax(1)]
    R = dict(modelo="CNN pequena (3 bloques conv, 2 cabezas)", t_ajuste_s=round(t_fit, 1), ms_por_imagen=round(t_inf, 3),
             parametros=int(m.count_params()), epocas=epochs)
    if p != "D":
        err = circ_err(est, ang[te]); ok = pred == obj[te]
        R.update(acc_clase=float(ok.mean()), mae_angulo=float(err.mean()), pct_dentro_10=float((err <= TOL).mean() * 100),
                 agarre_seguro=float((ok & (err <= TOL)).mean() * 100))
    else:
        # confianza de conocidos: ultima particion intercalada del entrenamiento (vistas ya vistas; cota optimista) vs desconocidos
        R.update(conf_media_desconocidos=float(pc.max(1).mean()))
        kn = m.predict(Xp[tr][::3], batch_size=256, verbose=0)[0].max(1)
        R.update(auroc_conf=auroc(kn, pc.max(1)), nota="conocidos medidos sobre vistas de entrenamiento (cota optimista)")
    return R


if __name__ == "__main__":
    which = sys.argv[1]
    out = {}
    for p in sys.argv[2:]:
        out[p] = (nn_baseline if which == "nn" else cnn_ref)(p)
        print(p, json.dumps(out[p]), flush=True)
    json.dump(out, open(f"results/baseline_{which}_{''.join(sys.argv[2:])}.json", "w"), indent=1)
