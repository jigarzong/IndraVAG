"""Registra en MLflow las ejecuciones ya calculadas (parametros, metricas, huella del dataset, artefactos)."""
import json, os, glob, numpy as np, mlflow
sha = str(np.load("data/cache/coil100.npz")["sha"])
mlflow.set_tracking_uri("sqlite:///mlflow.db"); mlflow.set_experiment("reto2-coil100")
def flat(prefix, d):
    for k, v in d.items():
        if isinstance(v, (int, float)) and v is not None: yield f"{prefix}{k}", float(v)
for fs, sfx in (("hogcolor", ""), ("pix", "_pix")):
    for p in "ABCD":
        R = json.load(open(f"results/{p}{sfx}.json"))
        with mlflow.start_run(run_name=f"protocolo_{p}_{fs}"):
            mlflow.set_tags(dict(protocolo=p, descriptores=fs, dataset_sha256=sha, seed=42))
            mlflow.log_params(dict(mejor_modelo=R["mejor_modelo"], n_train=R["n_train"], n_test=R["n_test"], pca=300,
                                   **{f"ang_{k}": v for k, v in R["angulo_params"].items()},
                                   **{f"umbral_{k}": round(v, 4) for k, v in R["umbrales"].items()},
                                   **{f"hp_{r['modelo']}": json.dumps(r["mejores_params"]) for r in R["comparacion_cv"] if r["modelo"] == R["mejor_modelo"]}))
            for r in R["comparacion_cv"]:
                mlflow.log_metric(f"cv_acc_{r['modelo'].replace(' ', '_')}", r["cv_acc"])
            for k, v in flat("test_", R["test"]): mlflow.log_metric(k, v)
            for k, v in flat("val_", {k: v for k, v in R["validacion_umbrales"].items() if v is not None}): mlflow.log_metric(k, v)
            mlflow.log_artifact(f"results/{p}{sfx}.json")
print("ok", len(glob.glob("mlruns/*/*")))
for nombre in ("nn", "cnn"):
    B = json.load(open(f"results/baseline_{nombre}_ABCD.json"))
    for p in "ABCD":
        with mlflow.start_run(run_name=f"referencia_{nombre}_{p}"):
            mlflow.set_tags(dict(protocolo=p, referencia=nombre, dataset_sha256=sha, seed=42))
            mlflow.log_params({k: v for k, v in B[p].items() if isinstance(v, (str, int)) and k not in ("nota",)})
            for k, v in B[p].items():
                if isinstance(v, float): mlflow.log_metric(k, v)
print("referencias registradas")
