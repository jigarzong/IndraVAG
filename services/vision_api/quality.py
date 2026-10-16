"""Validacion de calidad de imagen (determinista, sin LLM).

Los umbrales por defecto son heuristicas con margen respecto del rango observado en COIL-100
(muestra de 1.000 imagenes, semilla 42): varianza del laplaciano 46-1862, brillo medio 34-125,
desviacion estandar de gris 19-82. Quedan fuera del rango observado a proposito: una imagen
fuera de ese rango no se considera confiable aunque el modelo devuelva una clase.
Son ajustables por entorno y NO fueron validadas con imagenes reales de planta.
"""
import cv2
import numpy as np


def assess(rgb, side_before, cfg):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    m = dict(laplacian_var=float(cv2.Laplacian(g, cv2.CV_64F).var()), mean_gray=float(g.mean()), std_gray=float(g.std()),
             min_side_px=int(side_before))
    flags = []
    if side_before < cfg.q_min_side:
        flags.append("TOO_SMALL")
    if m["laplacian_var"] < cfg.q_blur_min:
        flags.append("BLURRY")
    if m["mean_gray"] < cfg.q_dark_max_mean:
        flags.append("TOO_DARK")
    if m["mean_gray"] > cfg.q_bright_min_mean:
        flags.append("OVEREXPOSED")
    if m["std_gray"] < cfg.q_contrast_min_std:
        flags.append("LOW_CONTRAST")
    return dict(ok=not flags, flags=flags, metrics={k: round(v, 3) for k, v in m.items()})
