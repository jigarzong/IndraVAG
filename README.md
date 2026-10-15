# Prueba Técnica: Líder Técnico de IA (Indra)

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
2. Colocar COIL-100 en `data/COIL-100` y ejecutar `INDRA_DATA=data/COIL-100 python src/prepare_data.py` (debe imprimir la huella `26635cc67c13f12b4010941d0e9e9af0bfda30c52007040377be0caba01c4c4a`).
3. `FS=hogcolor python src/run_protocols.py A B C D` y `FS=pix python src/run_protocols.py A B C D` (≈1 h en 2 núcleos). Referencias: `python src/baselines.py nn A B C D` y `python src/baselines.py cnn A B C D`.
4. `python src/analysis.py` (figuras y análisis de errores) y `python src/log_mlflow.py` (registro en MLflow, backend SQLite).
5. Notebook: ejecutar desde `notebooks/`; con `RERUN = False` carga los resultados guardados.

Semilla 42 en todos los pasos. Las secciones 4–8 del notebook son el primer intento (conservadas como historial) y sus cifras están reemplazadas por la sección 9.
