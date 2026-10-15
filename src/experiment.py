"""Reto 2 - experimento por protocolo (se copia a las celdas del notebook)."""
import numpy as np, time, json, warnings
warnings.filterwarnings("ignore")
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.isotonic import IsotonicRegression
from sklearn.base import clone
from lib import *

N_PCA = 300
TARGET_CLASS_ACC = 0.99      # acierto minimo de clase entre los aceptados (validacion)
TARGET_ANGLE_ACC = 0.95      # fraccion minima dentro de +-10 grados entre los aceptados (validacion)
TOL = 10.0                   # tolerancia de agarre del robot (supuesto del enunciado)
TAU_GRID = [0.003, 0.01, 0.03, 0.1]
EMB_GRID = ["pca", "raw"]   # espacio para estimar angulo: proyeccion del modelo o similitud directa (sin ajuste)
K_GRID = [2, 3, None]


def view_folds(p, ang_tr):
    """Validacion SOLO con vistas de entrenamiento, con la misma geometria de la prueba del protocolo:
    A intercalado (alternas), B sectores de 60 grados, C intercalado cada 3 vistas, D intercalado cada 3 vistas."""
    fid = {"A": (ang_tr // 10) % 2, "B": ang_tr // 60, "C": (ang_tr // 30) % 3, "D": (ang_tr // 5) % 3}[p]
    return [(np.where(fid != f)[0], np.where(fid == f)[0]) for f in np.unique(fid)]


def object_folds(y, n=5):
    """Objetos retenidos completos: simula objetos desconocidos usando solo el entrenamiento."""
    from sklearn.model_selection import GroupKFold
    return list(GroupKFold(n).split(y, y, groups=y))


def candidates():
    pre = lambda: [StandardScaler(), PCA(N_PCA, random_state=SEED)]
    return {
        "SVM-RBF": (make_pipeline(*pre(), SVC(kernel="rbf")),
                    {"svc__C": [10, 100], "svc__gamma": [3e-4, 6e-4, 1.2e-3]}),
        "Regresion logistica": (make_pipeline(*pre(), LogisticRegression(max_iter=500)),
                                {"logisticregression__C": [0.1, 1, 10]}),
        "Random Forest": (make_pipeline(*pre(), RandomForestClassifier(150, n_jobs=2, random_state=SEED)),
                          {"randomforestclassifier__max_features": ["sqrt"]}),
        "KNN": (make_pipeline(*pre(), KNeighborsClassifier(metric="cosine", weights="distance")),
                {"kneighborsclassifier__n_neighbors": [1, 3, 5]}),
    }


def select_classifier(X, y, cv):
    """Compara los 4 algoritmos con validacion cruzada estratificada SOLO sobre vistas de entrenamiento."""
    table, fitted = [], {}
    for name, (pipe, grid) in candidates().items():
        t = time.time()
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=2, refit=True).fit(X, y)
        i = gs.best_index_
        table.append(dict(modelo=name, cv_acc=float(gs.cv_results_["mean_test_score"][i]),
                          cv_std=float(gs.cv_results_["std_test_score"][i]),
                          mejores_params={k.split("__")[1]: (v if not isinstance(v, np.generic) else v.item()) for k, v in gs.best_params_.items()},
                          segundos=round(time.time() - t, 1)))
        fitted[name] = gs.best_estimator_
    best = max(table, key=lambda r: (round(r["cv_acc"], 4), -r["segundos"]))["modelo"]
    return table, best, fitted


def with_proba(pipe):
    p = clone(pipe)
    if isinstance(p[-1], SVC):
        p[-1].set_params(probability=True, random_state=SEED)
    return p


def embed(pipe, X):
    """Proyeccion normalizada (escalado + PCA del pipeline ya ajustado) usada para angulo y top-k."""
    return l2n(pipe[:-1].transform(X))


def oof_stage(pipe, X, y, ang, cv):
    """Predicciones fuera de muestra (OOF) sobre las vistas de entrenamiento: clase, confianza y datos para angulo."""
    n = len(y)
    pred, conf = np.zeros(n, int), np.zeros(n)
    folds = []
    for tr, va in cv:
        p = with_proba(pipe).fit(X[tr], y[tr])
        pr = p.predict_proba(X[va])
        pred[va] = p.classes_[pr.argmax(1)]
        conf[va] = pr.max(1)
        folds.append((tr, va, embed(p, X[tr]), embed(p, X[va])))
    return pred, conf, folds


def raw_embed(X):
    """Embedding sin parametros aprendidos (centrado por imagen + norma L2): no hay fuga posible."""
    return l2n(X - X.mean(1, keepdims=True))


def oof_angles(folds, pred, y, ang, tau, k, emb="pca", Xraw=None):
    n = len(y)
    est, unc, dmin = np.zeros(n), np.zeros(n), np.zeros(n)
    for tr, va, Ztr, Zva in folds:
        if emb == "raw":
            Ztr, Zva = Xraw[tr], Xraw[va]
        e, u, d = angle_posterior(Zva, Ztr, ang[tr], y[tr], pred[va], tau, k)
        est[va], unc[va], dmin[va] = e, u, d
    return est, unc, dmin


def simulate_unknowns(pipe, X, y, ang, tau, k, emb="pca"):
    """Objetos retenidos completos (GroupKFold por objeto) usando SOLO entrenamiento:
    puntajes (confianza, incertidumbre, distancia) que produciria un objeto desconocido."""
    n = len(y)
    conf, unc, dmin = np.zeros(n), np.zeros(n), np.zeros(n)
    for tr_, va in object_folds(y, 5):
        p = with_proba(pipe).fit(X[tr_], y[tr_])
        pr = p.predict_proba(X[va])
        cls = p.classes_[pr.argmax(1)]
        Zq, Zt = (raw_embed(X[va]), raw_embed(X[tr_])) if emb == "raw" else (embed(p, X[va]), embed(p, X[tr_]))
        e, u, d = angle_posterior(Zq, Zt, ang[tr_], y[tr_], cls, tau, k)
        conf[va], unc[va], dmin[va] = pr.max(1), u, d
    return conf, unc, dmin


def choose_thresholds(oof_conf, oof_ok, unc_o, dmin_o, err_o, u_conf, u_unc, u_dmin):
    """Umbrales por objetivos explicitos, todo en validacion (sin prueba):
    1) t_c: minimo umbral de confianza con acierto de clase >= 99% entre aceptados Y >= percentil 95 de la
       confianza de objetos desconocidos simulados (<=5% de falsos aceptados solo por clase).
    2) (t_u, t_d): rejilla por cuantiles; maximiza cobertura con P(clase ok y |error|<=10) >= 95% entre aceptados
       y falsa aceptacion de desconocidos simulados <= 5% con la regla combinada."""
    t_c1 = pick_threshold_max_coverage(oof_conf, oof_ok, TARGET_CLASS_ACC)
    t_c1 = oof_conf.max() + 1e-9 if t_c1 is None else t_c1
    t_c = float(max(t_c1, np.percentile(u_conf, 95)))
    safe = oof_ok & (err_o <= TOL)
    base = oof_conf >= t_c
    qs = np.r_[np.linspace(0.3, 1.0, 36)]
    cu = np.r_[np.quantile(unc_o[base], qs) if base.any() else [0.0], np.inf]
    cd = np.r_[np.quantile(dmin_o[base], qs) if base.any() else [0.0], np.inf]
    best = None
    for tu in np.unique(cu):
        for td in np.unique(cd):
            acc = base & (unc_o <= tu) & (dmin_o <= td)
            if not acc.any():
                continue
            s = safe[acc].mean()
            far = ((u_conf >= t_c) & (u_unc <= tu) & (u_dmin <= td)).mean()
            if s >= TARGET_ANGLE_ACC and far <= 0.05:
                cov = acc.mean()
                if best is None or cov > best[0] + 1e-12:
                    best = (cov, float(tu), float(td))
    if best is None:   # ningun umbral cumple los objetivos: se rechaza todo (decision conservadora)
        return t_c, -1.0, -1.0, 0.0
    return t_c, best[1], best[2], best[0]


def run_protocol(p, X, obj, ang, P, log=print, mlflow_run=None):
    tr, te = P[p]
    Xtr, ytr, atr = X[tr], obj[tr], ang[tr]
    cv = view_folds(p, atr)
    R = dict(protocolo=p, n_train=int(len(tr)), n_test=int(len(te)), validacion=f"{len(cv)} pliegues geometricos del protocolo (solo entrenamiento)")
    t0 = time.time()
    # 1) comparacion de algoritmos con validacion de misma geometria que la prueba (sin tocar prueba)
    table, best, fitted = select_classifier(Xtr, ytr, cv)
    R["comparacion_cv"], R["mejor_modelo"] = table, best
    log(f"[{p}] CV: " + " | ".join(f"{r['modelo']} {r['cv_acc']:.4f}" for r in table) + f" -> {best}")
    # 2) modelo final con probabilidades + OOF
    final = with_proba(fitted[best]).fit(Xtr, ytr)
    oof_pred, oof_conf, folds = oof_stage(fitted[best], Xtr, ytr, atr, cv)
    oof_ok = oof_pred == ytr
    R["oof_acc_clase"] = float(oof_ok.mean())
    # 3) angulo: eleccion de (tau, k) por error circular OOF
    Xraw = raw_embed(Xtr)
    best_ang = None
    for emb in EMB_GRID:
        for tau in TAU_GRID:
            for k in K_GRID:
                est, unc, dmin = oof_angles(folds, oof_pred, ytr, atr, tau, k, emb, Xraw)
                m = float(circ_err(est, atr).mean())
                if best_ang is None or m < best_ang[0]:
                    best_ang = (m, tau, k, emb)
    _, tau, k, emb = best_ang
    est_o, unc_o, dmin_o = oof_angles(folds, oof_pred, ytr, atr, tau, k, emb, Xraw)
    err_o = circ_err(est_o, atr)
    R["angulo_params"] = dict(tau=tau, k=k, espacio=emb)
    R["oof_mae_angulo"] = float(err_o.mean()); R["oof_pct_10"] = float((err_o <= TOL).mean() * 100)
    R["oof_spearman_incert_error"] = float(__import__("scipy.stats").stats.spearmanr(unc_o, err_o)[0])
    # 4) desconocidos simulados con objetos retenidos del entrenamiento + calibracion + umbrales
    u_conf, u_unc, u_dmin = simulate_unknowns(fitted[best], Xtr, ytr, atr, tau, k, emb)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof_conf, oof_ok.astype(float))
    R["oof_ece_cruda"] = ece_score(oof_conf, oof_ok)
    t_c, t_u, t_d, cov_val = choose_thresholds(oof_conf, oof_ok, unc_o, dmin_o, err_o, u_conf, u_unc, u_dmin)
    R["umbrales"] = dict(confianza_clase_cruda=t_c, incertidumbre_angulo_deg=t_u, distancia_vista_max=t_d)
    acc_oof = (oof_conf >= t_c) & (unc_o <= t_u) & (dmin_o <= t_d)
    R["validacion_umbrales"] = dict(
        cobertura=float(acc_oof.mean()),
        acierto_clase_aceptados=float(oof_ok[acc_oof].mean() * 100) if acc_oof.any() else None,
        seguro_aceptados=float(((oof_ok & (err_o <= TOL))[acc_oof]).mean() * 100) if acc_oof.any() else None,
        falsa_aceptacion_desconocidos_sim=float(((u_conf >= t_c) & (u_unc <= t_u) & (u_dmin <= t_d)).mean() * 100))
    log(f"[{p}] umbrales {R['umbrales']} validacion {R['validacion_umbrales']}")
    # 5) evaluacion UNICA en prueba
    Xte, yte, ate = X[te], obj[te], ang[te]
    pr = final.predict_proba(Xte)
    cls_pred = final.classes_[pr.argmax(1)]
    conf = pr.max(1)
    conf_cal = iso.predict(conf)
    Ztr, Zte = (raw_embed(Xtr), raw_embed(Xte)) if emb == "raw" else (embed(final, Xtr), embed(final, Xte))
    est, unc, dmin = angle_posterior(Zte, Ztr, atr, ytr, cls_pred, tau, k)
    accepted = (conf >= t_c) & (unc <= t_u) & (dmin <= t_d)
    sim = Zte @ Ztr.T
    top5 = np.argsort(-sim, axis=1)[:, :5]
    R["t_total_s"] = round(time.time() - t0, 1)
    out = dict(true_obj=yte, true_ang=ate, pred_obj=cls_pred, conf=conf, conf_cal=conf_cal, ang_est=est, unc=unc, dmin=dmin,
               accepted=accepted, top5=tr[top5], test_idx=te, oof_conf=oof_conf, oof_ok=oof_ok, oof_unc=unc_o, oof_err=err_o,
               oof_dmin=dmin_o, sim_conf=u_conf, sim_unc=u_unc, sim_dmin=u_dmin)
    if p != "D":
        from scipy.stats import spearmanr
        ok = cls_pred == yte
        err = circ_err(est, ate)
        safe = ok & (err <= TOL)
        R["test"] = dict(
            acc_clase=float(ok.mean()),
            mae_angulo=float(err.mean()), mediana_angulo=float(np.median(err)), pct_dentro_10=float((err <= TOL).mean() * 100),
            ece_cruda=ece_score(conf, ok), ece_calibrada=ece_score(conf_cal, ok),
            cobertura=float(accepted.mean()),
            acc_clase_aceptados=float(ok[accepted].mean() * 100) if accepted.any() else None,
            pct_10_aceptados=float((err[accepted] <= TOL).mean() * 100) if accepted.any() else None,
            agarre_seguro_aceptados=float(safe[accepted].mean() * 100) if accepted.any() else None,
            agarre_seguro_sin_rechazo=float(safe.mean() * 100),
            precision_at5=float((obj[tr[top5]] == yte[:, None]).mean()),
            spearman_incert_error=float(spearmanr(unc, err)[0]),
            auroc_deteccion_fallo=auroc(-unc[safe], -unc[~safe]) if (safe.any() and (~safe).any()) else None)
    else:
        R["test"] = dict(
            auroc_conf=auroc(oof_conf, conf),
            auroc_conf_simulado=auroc(oof_conf, u_conf),
            rechazo_desconocidos_solo_clase=float((conf < t_c).mean() * 100),
            rechazo_desconocidos_combinado=float((~accepted).mean() * 100),
            falso_rechazo_conocidos_oof=float((~acc_oof).mean() * 100),
            aceptados_desconocidos=float(accepted.mean() * 100),
            auroc_dmin=auroc(-dmin_o, -dmin))
    if mlflow_run is not None:
        mlflow_run(p, R, final, out)
    log(f"[{p}] listo en {R['t_total_s']} s")
    return R, out, final, iso
