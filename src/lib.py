"""Reto 2 - utilidades compartidas (se copian a las celdas del notebook).

Todo lo que ajusta parametros (escalado, PCA, hiperparametros, calibracion, umbrales)
usa unicamente las vistas de entrenamiento de cada protocolo.
"""
import numpy as np, cv2, time
from skimage.feature import hog

SEED = 42


# ---------------------------------------------------------------- features
def hog_features(imgs):
    """HOG sobre gris 64x64 (misma configuracion del notebook original)."""
    out = []
    for im in imgs:
        g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA)
        out.append(hog(g, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys"))
    return np.asarray(out, dtype=np.float32)


def color_features(imgs):
    """Histogramas HSV (H16,S8,V8) normalizados: 32 dims."""
    out = []
    for im in imgs:
        hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
        h = np.concatenate([
            np.histogram(hsv[..., 0], bins=16, range=(0, 180))[0],
            np.histogram(hsv[..., 1], bins=8, range=(0, 256))[0],
            np.histogram(hsv[..., 2], bins=8, range=(0, 256))[0]]).astype(np.float32)
        out.append(h / h.sum())
    return np.asarray(out, dtype=np.float32)


# ---------------------------------------------------------------- protocolos
def make_protocols(obj, ang, seed=SEED):
    """Devuelve {protocolo: (idx_train, idx_test)} sobre el arreglo completo de 7200 vistas."""
    idx = np.arange(len(obj))
    P = {}
    P["A"] = (idx[ang % 10 == 0], idx[ang % 10 != 0])
    rng = np.random.RandomState(seed)
    arc_start = {o: int(rng.choice(np.arange(0, 360, 5))) for o in np.unique(obj)}   # arco de 60 grados (12 vistas) por objeto
    blind = np.zeros(len(obj), bool)
    for o, s in arc_start.items():
        arc = [(s + 5 * k) % 360 for k in range(12)]
        blind |= (obj == o) & np.isin(ang, arc)
    P["B"] = (idx[~blind], idx[blind])
    P["C"] = (idx[ang % 30 == 0], idx[ang % 30 != 0])
    np.random.seed(seed)
    unknown = np.random.choice(np.unique(obj), size=10, replace=False)   # misma seleccion del notebook original
    isu = np.isin(obj, unknown)
    P["D"] = (idx[~isu], idx[isu])
    return P, arc_start, unknown


# ---------------------------------------------------------------- angulos
def circ_err(a, b):
    d = np.abs(np.asarray(a, float) - np.asarray(b, float)) % 360
    return np.minimum(d, 360 - d)


def circ_mean_deg(angles, w):
    r = np.deg2rad(angles)
    c, s = (w * np.cos(r)).sum(-1), (w * np.sin(r)).sum(-1)
    return np.rad2deg(np.arctan2(s, c)) % 360, np.sqrt(c ** 2 + s ** 2) / np.maximum(w.sum(-1), 1e-12)


def circ_std_deg(R):
    """Desviacion estandar circular (grados) a partir de la longitud resultante R."""
    R = np.clip(R, 1e-6, 1 - 1e-9)
    return np.rad2deg(np.sqrt(-2 * np.log(R)))


def l2n(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def angle_posterior(Z_query, Z_train, ang_train, obj_train, cls_pred, tau, k=None):
    """Estimacion de angulo condicionada a la clase predicha.

    Posterior sobre las vistas de entrenamiento de la clase predicha: softmax(sim/tau).
    Angulo = media circular ponderada; incertidumbre = desviacion circular (grados).
    Retorna (angulo, incertidumbre_deg, distancia_a_la_vista_mas_cercana).
    """
    n = len(Z_query)
    est, unc, dmin = np.zeros(n), np.zeros(n), np.zeros(n)
    for c in np.unique(cls_pred):
        qi = np.where(cls_pred == c)[0]
        ti = np.where(obj_train == c)[0]
        S = Z_query[qi] @ Z_train[ti].T            # similitud coseno (Z ya normalizado)
        a = ang_train[ti]
        if k is not None and S.shape[1] > k:
            top = np.argsort(-S, axis=1)[:, :k]
            mask = np.full_like(S, -np.inf)
            np.put_along_axis(mask, top, np.take_along_axis(S, top, 1), 1)
            S = mask
        W = np.exp((S - S.max(1, keepdims=True)) / tau)
        W /= W.sum(1, keepdims=True)
        m, R = circ_mean_deg(a[None, :].repeat(len(qi), 0), W)
        est[qi], unc[qi] = m, circ_std_deg(R)
        dmin[qi] = 1 - S.max(1)
    return est, unc, dmin


# ---------------------------------------------------------------- calibracion y metricas
def ece_score(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def auroc(pos_scores, neg_scores):
    """AUROC: probabilidad de que un positivo tenga mayor puntaje que un negativo (rangos)."""
    from scipy.stats import rankdata
    s = np.concatenate([pos_scores, neg_scores])
    r = rankdata(s)
    n1, n0 = len(pos_scores), len(neg_scores)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pick_threshold_max_coverage(score, ok, target):
    """Menor umbral t (aceptar score >= t) tal que la tasa de acierto entre aceptados sea >= target.
    Retorna el umbral que maximiza la cobertura cumpliendo el objetivo (o None si no existe)."""
    order = np.argsort(-score)
    s, o = score[order], ok[order].astype(float)
    acc = np.cumsum(o) / np.arange(1, len(o) + 1)
    valid = np.where(acc >= target)[0]
    if len(valid) == 0:
        return None
    j = valid.max()
    return float(s[j])
