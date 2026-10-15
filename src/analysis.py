"""Reto 2 - analisis de errores y figuras para el reporte (usa solo las salidas guardadas)."""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from lib import *

TOL = 10.0
os.makedirs("results/fig", exist_ok=True)
d = np.load("data/cache/coil100.npz")
imgs, obj, ang = d["imgs"], d["obj"], d["ang"]
NAVY, BLUE, ORANGE, GREEN, RED, GREY = "#14284b", "#1f5fa8", "#e07b00", "#2e8b57", "#c0392b", "#8a8f98"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})


def load(p, fs):
    sfx = "" if fs == "hogcolor" else f"_{fs}"
    return json.load(open(f"results/{p}{sfx}.json")), dict(np.load(f"results/{p}{sfx}_out.npz"))


def error_stats(o):
    err = circ_err(o["ang_est"], o["true_ang"])
    ok = o["pred_obj"] == o["true_obj"]
    s = {}
    wrong = ~ok
    pairs = Counter(zip(o["true_obj"][wrong].tolist(), o["pred_obj"][wrong].tolist()))
    s["top_confusiones"] = [dict(real=a, predicho=b, n=n) for (a, b), n in pairs.most_common(8)]
    per_obj = {int(c): float((~ok[o["true_obj"] == c]).mean() * 100) for c in np.unique(o["true_obj"])}
    s["objetos_peor_clase"] = sorted(per_obj.items(), key=lambda t: -t[1])[:8]
    per_obj_ang = {int(c): float(err[(o["true_obj"] == c) & ok].mean()) if ((o["true_obj"] == c) & ok).any() else np.nan for c in np.unique(o["true_obj"])}
    s["objetos_peor_angulo"] = sorted([(k, v) for k, v in per_obj_ang.items() if v == v], key=lambda t: -t[1])[:8]
    big = err > 90
    s["pct_error_mayor_90"] = float(big.mean() * 100)
    near180 = (err > 150)
    s["pct_error_cerca_180"] = float(near180.mean() * 100)
    s["pct_error_cerca_180_entre_errores_gt_10"] = float(near180[err > TOL].mean() * 100) if (err > TOL).any() else 0.0
    s["pct_fallo_clase_con_angulo_ok"] = float(((~ok) & (err <= TOL)).mean() * 100)
    # error angular por distancia a la vista de entrenamiento mas cercana (en grados): geometria del protocolo
    return s


def fig_error_dist(res):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    cols = dict(A=BLUE, B=RED, C=ORANGE)
    for p, (R, o) in res.items():
        if p == "D":
            continue
        err = np.sort(circ_err(o["ang_est"], o["true_ang"]))
        ax[0].plot(err, np.arange(1, len(err) + 1) / len(err) * 100, color=cols[p], label=f"Protocolo {p}")
    ax[0].axvline(TOL, color=GREY, ls="--", lw=1); ax[0].text(TOL + 2, 5, "±10°", color=GREY)
    ax[0].set_xlabel("Error angular circular (°)"); ax[0].set_ylabel("% de imágenes con error ≤ x"); ax[0].set_xlim(0, 180)
    ax[0].set_title("Distribución acumulada del error angular", fontsize=9, color=NAVY); ax[0].legend(frameon=False)
    for p, (R, o) in res.items():
        if p == "D":
            continue
        a = o["true_ang"]; err = circ_err(o["ang_est"], a)
        bins = np.arange(0, 361, 30)
        m = [err[(a >= lo) & (a < hi)].mean() for lo, hi in zip(bins[:-1], bins[1:])]
        ax[1].plot(bins[:-1] + 15, m, color=cols[p], marker="o", ms=3, label=f"Protocolo {p}")
    ax[1].set_xlabel("Ángulo real (°)"); ax[1].set_ylabel("Error angular medio (°)"); ax[1].set_xticks(range(0, 361, 60))
    ax[1].set_title("Error medio según el ángulo real", fontsize=9, color=NAVY)
    fig.tight_layout(); fig.savefig("results/fig/error_dist.png", dpi=200); plt.close(fig)


def fig_reliability(res):
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
    for a, p in zip(ax, "ABC"):
        R, o = res[p]
        ok = (o["pred_obj"] == o["true_obj"]).astype(float)
        for key, c, lab in (("conf", RED, "cruda"), ("conf_cal", GREEN, "calibrada")):
            edges = np.linspace(0, 1, 11); xs, ys = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (o[key] > lo) & (o[key] <= hi)
                if m.sum() >= 15:
                    xs.append(o[key][m].mean()); ys.append(ok[m].mean())
            a.plot(xs, ys, marker="o", ms=3, color=c, label=lab)
        a.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1)
        a.set_title(f"Protocolo {p}  (ECE {R['test']['ece_cruda']:.2f} → {R['test']['ece_calibrada']:.2f})", fontsize=8.5, color=NAVY)
        a.set_xlabel("Confianza media del bin")
    ax[0].set_ylabel("Acierto observado"); ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig("results/fig/reliability.png", dpi=200); plt.close(fig)


def fig_risk_coverage(res):
    """Diagnostico (no se usa para decidir umbrales): tasa de agarre seguro vs cobertura ordenando por incertidumbre."""
    fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.8))
    cols = dict(A=BLUE, B=RED, C=ORANGE)
    for p, (R, o) in res.items():
        if p == "D":
            continue
        safe = (o["pred_obj"] == o["true_obj"]) & (circ_err(o["ang_est"], o["true_ang"]) <= TOL)
        order = np.argsort(o["unc"] + 1e-3 * o["dmin"])
        cum = np.cumsum(safe[order]) / np.arange(1, len(order) + 1) * 100
        cov = np.arange(1, len(order) + 1) / len(order) * 100
        ax.plot(cov, cum, color=cols[p], label=f"Protocolo {p}")
        ax.plot([R["test"]["cobertura"] * 100], [R["test"]["agarre_seguro_aceptados"] or np.nan], marker="*", ms=9, color=cols[p], mec="black")
    ax.axhline(95, color=GREY, ls="--", lw=1)
    ax.set_xlabel("Cobertura (% aceptado)"); ax.set_ylabel("Agarres seguros entre aceptados (%)"); ax.set_ylim(0, 102)
    ax.set_title("Riesgo–cobertura (★ = umbral de validación)", fontsize=8.5, color=NAVY); ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig("results/fig/risk_coverage.png", dpi=200); plt.close(fig)


def fig_unknown(R, o):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5))
    bins = np.linspace(0, 1, 26)
    ax[0].hist(o["oof_conf"], bins, density=True, alpha=.6, color=BLUE, label="Conocidos (validación)")
    ax[0].hist(o["sim_conf"], bins, density=True, alpha=.6, color=ORANGE, label="Desconocidos simulados")
    ax[0].hist(o["conf"], bins, density=True, histtype="step", color=RED, lw=1.5, label="Desconocidos reales (prueba D)")
    ax[0].axvline(R["umbrales"]["confianza_clase_cruda"], color=GREY, ls="--"); ax[0].set_xlabel("Confianza de clase (SVM)"); ax[0].legend(frameon=False, fontsize=7)
    ax[0].set_title("Confianza: conocidos vs. desconocidos", fontsize=9, color=NAVY)
    ax[1].hist(o["oof_dmin"], 30, density=True, alpha=.6, color=BLUE, label="Conocidos")
    ax[1].hist(o["dmin"], 30, density=True, histtype="step", color=RED, lw=1.5, label="Desconocidos (D)")
    ax[1].set_xlabel("Distancia a la vista de entrenamiento más cercana"); ax[1].legend(frameon=False, fontsize=7)
    ax[1].set_title("Distancia al vecino más cercano", fontsize=9, color=NAVY)
    fig.tight_layout(); fig.savefig("results/fig/unknown.png", dpi=200); plt.close(fig)


def fig_gallery(p, R, o, n=6, name="gallery"):
    err = circ_err(o["ang_est"], o["true_ang"]); ok = o["pred_obj"] == o["true_obj"]
    idx_bad = np.where(~ok)[0]
    idx_bad = idx_bad[np.argsort(-o["conf"][idx_bad])]               # errores de clase MAS confiados (los peligrosos)
    _, u = np.unique(o["true_obj"][idx_bad], return_index=True); idx_bad = idx_bad[np.sort(u)][:n]
    idx_ang = np.where(ok & (err > 60))[0]
    idx_ang = idx_ang[np.argsort(-err[idx_ang])]                     # peores errores angulares con clase correcta
    _, u = np.unique(o["true_obj"][idx_ang], return_index=True); idx_ang = idx_ang[np.sort(u)][:n]
    fig, ax = plt.subplots(3, n, figsize=(7.2, 4.3))
    for j in range(n):
        for r in range(3):
            ax[r, j].axis("off")
        if j < len(idx_bad):
            i = idx_bad[j]; q = o["test_idx"][i]
            pred_im = np.where(obj == o["pred_obj"][i])[0][0]
            ax[0, j].imshow(imgs[q]); ax[0, j].set_title(f"real: obj {obj[q]}\n{ang[q]}°", fontsize=6.5)
            ax[1, j].imshow(imgs[pred_im]); ax[1, j].set_title(f"predicho: obj {o['pred_obj'][i]}\nconf {o['conf'][i]:.2f}", fontsize=6.5)
        if j < len(idx_ang):
            i = idx_ang[j]; q = o["test_idx"][i]
            cand = np.where((obj == o["true_obj"][i]) & (np.abs(((ang - o["ang_est"][i] + 180) % 360) - 180) < 3))[0]
            ax[2, j].imshow(imgs[q]); ax[2, j].set_title(f"obj {obj[q]}: real {ang[q]}°\nestimado {o['ang_est'][i]:.0f}°", fontsize=6.5)
    ax[0, 0].text(-0.15, 0.5, "Real\n(clase errada)", transform=ax[0, 0].transAxes, ha="right", va="center", fontsize=7)
    ax[1, 0].text(-0.15, 0.5, "Objeto\npredicho", transform=ax[1, 0].transAxes, ha="right", va="center", fontsize=7)
    ax[2, 0].text(-0.15, 0.5, "Peor error\nangular", transform=ax[2, 0].transAxes, ha="right", va="center", fontsize=7)
    fig.subplots_adjust(left=0.1, right=0.99, wspace=0.05, hspace=0.3)
    fig.savefig(f"results/fig/{name}.png", dpi=200); plt.close(fig)


def fig_models(allres):
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    names = ["SVM-RBF", "Regresion logistica", "Random Forest", "KNN"]
    w = 0.2
    for i, p in enumerate("ABCD"):
        R = allres[p][0]
        vals = [next(r["cv_acc"] for r in R["comparacion_cv"] if r["modelo"] == n) * 100 for n in names]
        ax.bar(np.arange(4) + (i - 1.5) * w, vals, w, label=f"Protocolo {p}", color=[BLUE, RED, ORANGE, GREEN][i])
    ax.set_xticks(range(4)); ax.set_xticklabels(["SVM-RBF", "Reg. logística", "Random Forest", "KNN (coseno)"])
    ax.set_ylabel("Acierto de clase en\nvalidación (%)"); ax.set_ylim(55, 112); ax.legend(frameon=False, ncol=4, fontsize=7.5, loc="upper center")
    ax.set_title("Comparación de algoritmos (validación solo con entrenamiento)", fontsize=9, color=NAVY)
    fig.tight_layout(); fig.savefig("results/fig/models.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    final = json.load(open("results/final_selection.json"))     # {"A": "hogcolor"|"pix", ...}
    res = {p: load(p, fs) for p, fs in final.items()}
    summ = {p: error_stats(res[p][1]) for p in res if p != "D"}
    json.dump(summ, open("results/error_analysis.json", "w"), indent=1)
    fig_error_dist(res); fig_reliability(res); fig_risk_coverage(res); fig_unknown(*res["D"]); fig_models(res)
    for p in "ABC":
        fig_gallery(p, *res[p], name=f"gallery_{p}")
    print(json.dumps(summ, indent=1))
