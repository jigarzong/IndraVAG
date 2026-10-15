# Model Card: Motor de Visión Indra Pick & Pack

## Detalles del modelo
* **Desarrollador:** José Ignacio Garzón Grajales (Líder Técnico de IA, prueba técnica Indra). **Fecha:** octubre de 2026.
* **Tipo:** pipeline de Machine Learning clásico: descriptores (HOG 64×64 en gris + histograma HSV 16/8/8, o píxeles 32×32 RGB) → StandardScaler → PCA(300) → SVM-RBF con probabilidades (Platt) + calibración isotónica.
  El ángulo se estima con un posterior sobre las vistas de entrenamiento de la clase predicha (media circular ponderada) y la recuperación Top-K usa similitud coseno.
* **Descriptores por protocolo (elegidos por validación, solo con entrenamiento):** A HOG + HSV, B Píxeles 32×32, C Píxeles 32×32, D HOG + HSV.
* **Versión de datos:** SHA-256 del dataset `26635cc67c13f12b4010941d0e9e9af0bfda30c52007040377be0caba01c4c4a`. Semilla 42. Ejecuciones registradas en MLflow (`mlflow.db`).

## Uso previsto
Identificar el SKU y estimar su ángulo (0°–360°) en una estación de pick & pack, entregando al robot una decisión `GRASP` / `RETRY_VIEW` / `REJECT` con confianza calibrada e incertidumbre angular.
La tolerancia mecánica supuesta es ±10°.

## Datos
COIL-100: 7.200 imágenes de 100 objetos, 72 vistas cada 5°, fondo negro, iluminación fija. No contiene datos personales.
Particiones: A (vistas cada 10° para entrenar), B (arco ciego de 60° por objeto), C (vistas cada 30°), D (10 objetos desconocidos). Todo ajuste usa solo entrenamiento; la prueba se evalúa una vez.

## Métricas (prueba)
| Protocolo | Clase | Error angular medio | Dentro de ±10° | Agarre seguro sin rechazo | Cobertura con rechazo | Seguro entre aceptados |
|---|---|---|---|---|---|---|
| A | 100.0% | 2.4° | 96.9% | 96.9% | 45.2% | 98.1% |
| B | 95.1% | 66.8° | 18.6% | 17.3% | 2.2% | 84.6% |
| C | 95.1% | 15.4° | 73.1% | 71.9% | 6.0% | 94.4% |

* **Protocolo D:** 99.6% de las vistas de objetos desconocidos se rechazan (AUROC de la confianza 0.95; de la distancia al vecino más cercano 0.96). La regla rechaza además el 76.4% de las vistas conocidas en validación.
* **Calibración (ECE en prueba, cruda → calibrada):** A 0.47→0.00, B 0.43→0.02, C 0.76→0.02.
* **Referencias:** un 1-NN sobre píxeles obtiene 100.0% / 89.8% / 97.4% de clase en A/B/C y 75.4% dentro de ±10° en C; una CNN pequeña sin ajuste obtiene 99.8% de clase en A pero 26.7% dentro de ±10° (cabeza angular sin ajustar).

## Qué se puede y qué no se puede afirmar
* El ángulo es fiable (±10°) con vistas cada 10° (A) y solo parcialmente con 30° (C). **En B (arco sin observar) el ángulo no es recuperable** con este enfoque; el sistema debe usar solo la clase y pedir otra vista o intervención humana.
* Los umbrales de rechazo se fijan en validación con objetivos explícitos (≥99 % de clase entre aceptados, ≥95 % de agarres seguros, ≤5 % de falsa aceptación de desconocidos simulados). Son conservadores porque la validación solo con entrenamiento ve una geometría más escasa que la prueba.
* En prueba el objetivo de ≥95 % de agarres seguros entre aceptados se cumple en A (98.1%), queda cerca en C (94.4%) y no se cumple en B (84.6% con 2.2% de cobertura).

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
