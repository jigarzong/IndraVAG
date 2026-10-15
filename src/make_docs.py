# -*- coding: utf-8 -*-
"""Genera model_card.md y README.md con las cifras reales de results/."""
import json, sys
fin = json.load(open("results/final_selection.json"))
ld = lambda p, fs: json.load(open(f"results/{p}{'' if fs == 'hogcolor' else '_' + fs}.json"))
R = {p: ld(p, fin[p]) for p in "ABCD"}
T = {p: R[p]["test"] for p in "ABC"}; D = R["D"]["test"]
nn = json.load(open("results/baseline_nn_ABCD.json")); cnn = json.load(open("results/baseline_cnn_ABCD.json"))
meta = json.load(open("results/meta.json"))
f = lambda x, d=1: "—" if x is None else f"{x:.{d}f}"
fsn = {"hogcolor": "HOG + HSV", "pix": "Píxeles 32×32"}

card = f"""# Model Card: Motor de Visión Indra Pick & Pack

## Detalles del modelo
* **Desarrollador:** José Ignacio Garzón Grajales (Líder Técnico de IA, prueba técnica Indra). **Fecha:** octubre de 2026.
* **Tipo:** pipeline de Machine Learning clásico: descriptores (HOG 64×64 en gris + histograma HSV 16/8/8, o píxeles 32×32 RGB) → StandardScaler → PCA(300) → SVM-RBF con probabilidades (Platt) + calibración isotónica.
  El ángulo se estima con un posterior sobre las vistas de entrenamiento de la clase predicha (media circular ponderada) y la recuperación Top-K usa similitud coseno.
* **Descriptores por protocolo (elegidos por validación, solo con entrenamiento):** A {fsn[fin['A']]}, B {fsn[fin['B']]}, C {fsn[fin['C']]}, D {fsn[fin['D']]}.
* **Versión de datos:** SHA-256 del dataset `{meta['sha']}`. Semilla 42. Ejecuciones registradas en MLflow (`mlflow.db`).

## Uso previsto
Identificar el SKU y estimar su ángulo (0°–360°) en una estación de pick & pack, entregando al robot una decisión `GRASP` / `RETRY_VIEW` / `REJECT` con confianza calibrada e incertidumbre angular.
La tolerancia mecánica supuesta es ±10°.

## Datos
COIL-100: 7.200 imágenes de 100 objetos, 72 vistas cada 5°, fondo negro, iluminación fija. No contiene datos personales.
Particiones: A (vistas cada 10° para entrenar), B (arco ciego de 60° por objeto), C (vistas cada 30°), D (10 objetos desconocidos). Todo ajuste usa solo entrenamiento; la prueba se evalúa una vez.

## Métricas (prueba)
| Protocolo | Clase | Error angular medio | Dentro de ±10° | Agarre seguro sin rechazo | Cobertura con rechazo | Seguro entre aceptados |
|---|---|---|---|---|---|---|
""" + "\n".join(
    f"| {p} | {f(T[p]['acc_clase']*100)}% | {f(T[p]['mae_angulo'])}° | {f(T[p]['pct_dentro_10'])}% | {f(T[p]['agarre_seguro_sin_rechazo'])}% | {f(T[p]['cobertura']*100)}% | {f(T[p]['agarre_seguro_aceptados'])}% |"
    for p in "ABC") + f"""

* **Protocolo D:** {f(D['rechazo_desconocidos_combinado'])}% de las vistas de objetos desconocidos se rechazan (AUROC de la confianza {f(D['auroc_conf'],2)}; de la distancia al vecino más cercano {f(D['auroc_dmin'],2)}). La regla rechaza además el {f(D['falso_rechazo_conocidos_oof'])}% de las vistas conocidas en validación.
* **Calibración (ECE en prueba, cruda → calibrada):** A {T['A']['ece_cruda']:.2f}→{T['A']['ece_calibrada']:.2f}, B {T['B']['ece_cruda']:.2f}→{T['B']['ece_calibrada']:.2f}, C {T['C']['ece_cruda']:.2f}→{T['C']['ece_calibrada']:.2f}.
* **Referencias:** un 1-NN sobre píxeles obtiene {f(nn['A']['acc_clase']*100)}% / {f(nn['B']['acc_clase']*100)}% / {f(nn['C']['acc_clase']*100)}% de clase en A/B/C y {f(nn['C']['pct_dentro_10'])}% dentro de ±10° en C; una CNN pequeña sin ajuste obtiene {f(cnn['A']['acc_clase']*100)}% de clase en A pero {f(cnn['A']['pct_dentro_10'])}% dentro de ±10° (cabeza angular sin ajustar).

## Qué se puede y qué no se puede afirmar
* El ángulo es fiable (±10°) con vistas cada 10° (A) y solo parcialmente con 30° (C). **En B (arco sin observar) el ángulo no es recuperable** con este enfoque; el sistema debe usar solo la clase y pedir otra vista o intervención humana.
* Los umbrales de rechazo se fijan en validación con objetivos explícitos (≥99 % de clase entre aceptados, ≥95 % de agarres seguros, ≤5 % de falsa aceptación de desconocidos simulados). Son conservadores porque la validación solo con entrenamiento ve una geometría más escasa que la prueba.
* En prueba el objetivo de ≥95 % de agarres seguros entre aceptados se cumple en A ({f(T['A']['agarre_seguro_aceptados'])}%), queda cerca en C ({f(T['C']['agarre_seguro_aceptados'])}%) y no se cumple en B ({f(T['B']['agarre_seguro_aceptados'])}% con {f(T['B']['cobertura']*100)}% de cobertura).

## Limitaciones conocidas
* Condiciones de laboratorio: fondo negro, iluminación constante, sin oclusión, un solo plano de rotación. Las cifras no se extrapolan a la banda real sin una evaluación en planta.
* Objetos casi simétricos (cajas, frascos, tazas) son ambiguos a 180°.
* Una sola partición por protocolo (semilla 42), sin intervalos entre semillas. Los mejores hiperparámetros del SVM quedaron en el borde de la rejilla (C=100, γ=3·10⁻⁴).
* Los píxeles se añadieron como candidato tras observar la línea base en prueba; la elección entre descriptores se hizo por validación.

## Condiciones de no uso
* **Prohibido operar «a ciegas»:** el modelo no debe conectarse al brazo sin la regla de rechazo triple (confianza, incertidumbre angular, distancia al vecino más cercano). Un error fuera de ±10° puede dañar la pinza o el producto.
* No usar el ángulo en el escenario de arco sin observar (B) ni con menos de ~36 vistas por objeto sin revalidar los umbrales.
* No sustituir por una arquitectura profunda sin comparar costo, latencia y mantenimiento frente a esta línea base (el 1-NN sobre píxeles ya es competitivo en este dataset).

## Monitoreo recomendado
Tasa de rechazo y de reintentos, distribución de la distancia al vecino más cercano (deriva), ECE con etiquetas de auditoría, tasa de intervención humana y recalibración periódica.
"""
open(sys.argv[1] + "/model_card.md", "w").write(card)

readme = f"""# Prueba Técnica: Líder Técnico de IA (Indra)

Solución técnica de extremo a extremo para reemplazar tareas manuales en una línea de producción semiautomática con Visión Artificial, Machine Learning clásico y arquitecturas agénticas.

## Entregables

| Reto | Entregable | Ubicación |
| :--- | :--- | :--- |
| Reto 1 | Diagramas AS-IS y TO-BE (BPMN) | `reports/` (PDF del Reto 1) |
| Reto 2 | Notebook reproducible (CRISP-DM, secciones 9 en adelante vigentes) | [`notebooks/Indra_MLModel_Vision.ipynb`](./notebooks/Indra_MLModel_Vision.ipynb) |
| Reto 2 | Reporte de resultados por protocolo, análisis de errores y model card (PDF) | [`reports/Reto 2 - Resultados, analisis de errores y model card.pdf`](./reports) |
| Reto 2 | Model Card | [`model_card.md`](./model_card.md) |
| Reto 2 | Modelo exportado y función de inferencia | [`models/protocolo_A/`](./models/protocolo_A), [`src/inference.py`](./src/inference.py) |
| Reto 3 | Arquitectura agéntica, herramientas y diagramas | Documento del Reto 3 |
| Reto 4 | Despliegue, servicio y operación local | Pendiente |
| Reto 5 | Plan de proyecto, fases y RACI | Pendiente |
| Reto 6 | Arquitectura Cloud (AWS vs Azure) y FinOps | Pendiente |

## Estructura
```
src/        lib.py (descriptores, protocolos, ángulos), experiment.py (validación, umbrales, run_protocol),
            run_protocols.py, baselines.py, analysis.py, prepare_data.py, inference.py, export_model.py, log_mlflow.py
results/    JSON por protocolo, salidas de prueba, figuras y análisis de errores
models/     protocolo_A/bundle.joblib + thresholds.json
notebooks/  Indra_MLModel_Vision.ipynb
data/       COIL-100 (no versionado) y data/cache/ (generado)
```

## Reproducción (Reto 2)
1. `pip install numpy pandas matplotlib opencv-python scikit-learn scikit-image scipy joblib mlflow tensorflow-cpu nbformat`
2. Colocar COIL-100 en `data/COIL-100` y ejecutar `INDRA_DATA=data/COIL-100 python src/prepare_data.py` (debe imprimir la huella `{meta['sha']}`).
3. `FS=hogcolor python src/run_protocols.py A B C D` y `FS=pix python src/run_protocols.py A B C D` (≈1 h en 2 núcleos). Referencias: `python src/baselines.py nn A B C D` y `python src/baselines.py cnn A B C D`.
4. `python src/analysis.py` (figuras y análisis de errores) y `python src/log_mlflow.py` (registro en MLflow, backend SQLite).
5. Notebook: ejecutar desde `notebooks/`; con `RERUN = False` carga los resultados guardados.

Semilla 42 en todos los pasos. Las secciones 4–8 del notebook son el primer intento (conservadas como historial) y sus cifras están reemplazadas por la sección 9.
"""
open(sys.argv[1] + "/README.md", "w").write(readme)
print("ok")
