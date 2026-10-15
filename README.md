# Prueba Técnica: Líder Técnico de IA (Indra)

Este repositorio contiene la solución técnica completa de extremo a extremo para el reemplazo de tareas manuales en una línea de producción semiautomática, utilizando Visión Artificial, Machine Learning Clásico y Arquitecturas Agénticas, conforme a las rúbricas y requerimientos del negocio.

## Estructura de Entregables Obligatorios

| Reto | Entregable | Ubicación en el Repositorio |
| :--- | :--- | :--- |
| **Reto 1** | Diagramas AS-IS y TO-BE (BPMN) | [`reto1_indra_bpmn.pdf`](./reto1_indra_bpmn.pdf) |
| **Reto 2** | Notebook reproducible (Experimentación CRISP-DM) | [`Indra_MLModel_Vision.ipynb`](./Indra_MLModel_Vision.ipynb) |
| **Reto 2** | Resultados, análisis de errores y justificación de métricas | Incluidos dentro del Notebook |
| **Reto 2** | Model Card | [`model_card.md`](./model_card.md) |
| **Reto 3** | Arquitectura agéntica, herramientas (tools) y diagramas | Pendiente: Reto 3 |
| **Reto 4** | Despliegue, servicio y operación local | Pendiente: Reto 4 |
| **Reto 5** | Plan de proyecto, fases y RACI | Pendiente: Reto 5 |
| **Reto 6** | Arquitectura Cloud (AWS vs Azure) y Modelo FinOps | Pendiente: Reto 6 |

## Ejecución Local y Reproducibilidad (Reto 2)

Para reproducir los experimentos del modelo de ML clásico (SVM + HOG):

1. Clonar el repositorio: `git clone https://github.com/jigarzong/IndraVAG.git`
2. Instalar dependencias en el entorno virtual: `pip install numpy pandas matplotlib opencv-python scikit-learn scikit-image ipykernel`
3. Asegurarse de que el dataset COIL-100 esté descargado en `C:\IndraVAG\data\COIL-100`.
4. Ejecutar todas las celdas del archivo `Indra_MLModel_Vision.ipynb`. Las semillas están fijadas (`SEED=42`) para garantizar resultados idénticos a los reportados en la prueba.
