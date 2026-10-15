# Model Card: Motor de Visión Indra-Pick&Pack

## Detalles del Modelo
* **Desarrollador:** Equipo de IA (Líder Técnico de IA)
* **Fecha:** Octubre 2026
* **Tipo de Modelo:** Pipeline de Machine Learning Clásico (Extracción HOG + SVM Probabilístico y KNN para recuperación visual). No se utiliza Deep Learning en el flujo primario por requerimiento estricto de la prueba.

## Uso Previsto
Identificación de clases de productos de consumo masivo (SKUs) y estimación de su ángulo de rotación (0°-360°) a partir de imágenes en banda transportadora, para asistir operativamente a un brazo robótico de pick & pack.

## Datos de Entrenamiento
* **Dataset:** COIL-100 (Columbia Object Image Library). 
* **Características:** 7200 imágenes de 100 objetos aislados sobre fondo negro.
* **Preprocesamiento:** Redimensionamiento a 64x64, conversión a escala de grises y extracción de gradientes orientados (HOG).

## Métricas de Rendimiento (Protocolo A - Interpolación)
* **Exactitud de Identidad (Clasificación):** > 98% (SVM RBF)
* **Precisión Angular Mecánica (Agarres dentro de ±10°):** 54.67%
* *Nota:* Dado el acierto base del 54.67%, el sistema opera en conjunto con un mecanismo de **Rechazo Seguro**, evaluando la entropía (incertidumbre) del modelo para bloquear acciones dudosas.

## Limitaciones Conocidas
El modelo fue entrenado en condiciones de laboratorio estáticas (iluminación ideal, fondo negro continuo y sin oclusiones). Es altamente sensible a:
* Presencia de suciedad, brillos o texturas marcadas en la banda transportadora.
* Oclusiones parciales del objeto (cuando sale cortado del encuadre de la cámara).

## Condiciones de No-Uso
* **Prohibido el uso "A Ciegas":** Este modelo NO DEBE conectarse directamente a la ejecución física del brazo robótico sin el módulo de "Rechazo Seguro por Umbral de Incertidumbre" activado. Un fallo angular fuera de ±10° puede romper la pinza robótica o dañar el producto.
* **No apto para Deep Learning en primera fase:** El modelo no debe ser sustituido por una arquitectura CNN (ej. ResNet) en planta sin justificar previamente los costos de inferencia y la latencia introducida en el ciclo de empaque.