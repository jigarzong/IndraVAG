# -*- coding: utf-8 -*-
"""Genera el entregable PDF del Reto 2 con el estilo del PDF del Reto 1 (portada Indra, titulos serif, vinetas cuadradas, pie 'Page N')."""
import json, os
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle, Image,
                                PageBreak, KeepTogether, NextPageTemplate, Flowable)

FD = "/usr/share/fonts/truetype/crosextra/"
pdfmetrics.registerFont(TTFont("Serif", FD + "Caladea-Regular.ttf"))
pdfmetrics.registerFont(TTFont("Serif-B", FD + "Caladea-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Sans", FD + "Carlito-Regular.ttf"))
pdfmetrics.registerFont(TTFont("Sans-B", FD + "Carlito-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Sans-I", FD + "Carlito-Italic.ttf"))
pdfmetrics.registerFont(TTFont("DJ", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-B", italic="Sans-I", boldItalic="Sans-B")
pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-B", italic="Serif", boldItalic="Serif-B")

NAVY = colors.HexColor("#1b3a5c"); BLUE = colors.HexColor("#1f5fa8"); TXT = colors.HexColor("#2d3440")
GREYBG = colors.HexColor("#f4f6f8"); LINE = colors.HexColor("#d5dbe3"); COVER = colors.HexColor("#e1e4e6")
RED = colors.HexColor("#b03a2e"); GREEN = colors.HexColor("#2e7d4f")

FOOT = "Reto 2 - Resultados por protocolo, análisis de errores y model card"
W, H = A4
LM = RM = 48
FIG = "results/fig/"

st = dict(
    h1=ParagraphStyle("h1", fontName="Serif", fontSize=19, leading=23, textColor=NAVY, spaceAfter=4),
    h2=ParagraphStyle("h2", fontName="Serif-B", fontSize=12.5, leading=16, textColor=NAVY, spaceBefore=9, spaceAfter=4),
    p=ParagraphStyle("p", fontName="Sans", fontSize=9.3, leading=13.2, textColor=TXT, spaceAfter=5),
    b=ParagraphStyle("b", bulletFontName="DJ", bulletFontSize=5.5, fontName="Sans", fontSize=9.3, leading=13, textColor=TXT, leftIndent=12, bulletIndent=0, spaceAfter=3.5),
    cap=ParagraphStyle("cap", fontName="Sans-I", fontSize=8, leading=10.5, textColor=colors.HexColor("#5b6572"), spaceAfter=7),
    tc=ParagraphStyle("tc", fontName="Sans", fontSize=8.1, leading=10, textColor=TXT),
    th=ParagraphStyle("th", fontName="Sans-B", fontSize=8.1, leading=10, textColor=colors.white),
    note=ParagraphStyle("note", fontName="Sans", fontSize=8.4, leading=11.5, textColor=TXT),
)


def P(t, s="p"): return Paragraph(t, st[s])
def B(t): return Paragraph(t, st["b"], bulletText="■")


class Rule(Flowable):
    def __init__(self, w): super().__init__(); self.w = w
    def wrap(self, aw, ah): return self.w, 4
    def draw(self): self.canv.setStrokeColor(LINE); self.canv.setLineWidth(.8); self.canv.line(0, 1, self.w, 1)


def H1(t): return [Spacer(1, 12), Paragraph(t, st["h1"]), Rule(W - LM - RM), Spacer(1, 5)]
def H2(t): return Paragraph(t, st["h2"])


def img(name, width):
    from PIL import Image as PI
    w, h = PI.open(FIG + name).size
    return Image(FIG + name, width=width, height=width * h / w)


def panel(flow, pad=8):
    t = Table([[flow]], colWidths=[W - LM - RM])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), GREYBG), ("BOX", (0, 0), (-1, -1), .6, LINE),
                           ("ROUNDEDCORNERS", [5, 5, 5, 5]), ("LEFTPADDING", (0, 0), (-1, -1), pad), ("RIGHTPADDING", (0, 0), (-1, -1), pad),
                           ("TOPPADDING", (0, 0), (-1, -1), pad), ("BOTTOMPADDING", (0, 0), (-1, -1), pad)]))
    return t


def table(rows, widths, header=True, zebra=True, align_right_from=1, fs=None):
    data = []
    for i, r in enumerate(rows):
        data.append([Paragraph(str(c), st["th"] if (header and i == 0) else st["tc"]) for c in r])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    sty = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LINEBELOW", (0, 0), (-1, -1), .4, LINE),
           ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
           ("TOPPADDING", (0, 0), (-1, -1), 3.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2)]
    if header:
        sty.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    if zebra:
        for i in range(1, len(rows)):
            if i % 2 == 0: sty.append(("BACKGROUND", (0, i), (-1, i), GREYBG))
    t.setStyle(TableStyle(sty))
    return t


# ---------------------------------------------------------------- paginas
def cover(c, doc):
    c.saveState()
    c.setFillColor(COVER); c.setStrokeColor(colors.HexColor("#9aa3ad")); c.setLineWidth(.6)
    c.roundRect(LM - 4, 150, W - 2 * LM + 8, H - 250, 3, stroke=1, fill=1)
    c.setFillColor(colors.HexColor("#4a5568")); c.setFont("Serif", 31)
    c.drawString(LM + 42, H - 175, "Prueba Técnica Líder"); c.drawString(LM + 42, H - 214, "Técnico de IA")
    c.setFillColor(colors.HexColor("#1a5a86")); c.setFont("Serif", 10.5)
    c.drawString(LM + 42, 258, "Evaluación técnica")
    c.drawString(LM + 42, 237, "Candidato: José Ignacio Garzón Grajales")
    c.setFillColor(colors.HexColor("#444444")); c.setFont("Sans", 8)
    c.drawString(LM - 4, 100, "indracompany.com")
    c.setFillColor(colors.HexColor("#0b4a7a")); c.setFont("Sans-B", 9)
    c.drawRightString(W - LM + 4, 100, ">> INDRA GROUP")
    c.restoreState()


def footer(c, doc):
    c.saveState()
    c.setStrokeColor(LINE); c.setLineWidth(.8); c.line(LM, 42, W - LM, 42)
    c.setFont("Sans", 8.2); c.setFillColor(TXT)
    c.drawString(LM, 30, FOOT); c.drawRightString(W - LM, 30, f"Page {doc.page - 1}")
    c.restoreState()


def pct(x, d=1): return "—" if x is None else f"{x:.{d}f}%"
def f1(x, d=1): return "—" if x is None else f"{x:.{d}f}"


def build(out="Reto 2 - Resultados, analisis de errores y model card.pdf"):
    fin = json.load(open("results/final_selection.json"))
    def ld(p, fs): return json.load(open(f"results/{p}{'' if fs == 'hogcolor' else '_' + fs}.json"))
    R = {p: ld(p, fin[p]) for p in "ABCD"}
    alt = {p: ld(p, "pix" if fin[p] == "hogcolor" else "hogcolor") for p in "ABCD"}
    ea = json.load(open("results/error_analysis.json"))
    nn = json.load(open("results/baseline_nn_ABCD.json"))
    cnn = json.load(open("results/baseline_cnn_ABCD.json"))
    meta = json.load(open("results/meta.json"))
    fsname = {"hogcolor": "HOG + color HSV", "pix": "Píxeles 32×32"}
    T = {p: R[p]["test"] for p in "ABC"}

    doc = BaseDocTemplate(out, pagesize=A4, leftMargin=LM, rightMargin=RM, topMargin=48, bottomMargin=56,
                          title="Reto 2 - Resultados por protocolo, análisis de errores y model card",
                          author="José Ignacio Garzón Grajales")
    fr = Frame(LM, 56, W - LM - RM, H - 48 - 56, id="f", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate("cover", [Frame(0, 0, 1, 1)], onPage=cover),
                          PageTemplate("body", [fr], onPage=footer)])
    S = [NextPageTemplate("body"), PageBreak()]

    # ---------------------------------------------------------------- 6.1 resumen
    S += H1("6. Reto 2 – Visión clásica: resultados, errores y model card")
    S.append(P("Este reporte consolida los resultados de los cuatro protocolos obligatorios sobre COIL-100 (100 objetos × 72 vistas = 7.200 imágenes) para "
               "identificar el objeto, estimar su ángulo (tolerancia mecánica ±10°) y decidir si el robot puede actuar o debe rechazar. "
               "Todo ajuste (escalado, PCA, hiperparámetros, calibración y umbrales) usa únicamente las vistas de entrenamiento de cada protocolo; "
               "la partición de prueba se evalúa una sola vez."))
    S.append(H2("Resumen ejecutivo"))
    S.append(B(f"<b>A – Interpolación (36 vistas/objeto):</b> {pct(T['A']['acc_clase']*100)} de acierto de clase, error angular medio {f1(T['A']['mae_angulo'])}° y "
               f"{pct(T['A']['pct_dentro_10'])} de las imágenes dentro de ±10°. La regla de rechazo acepta {pct(T['A']['cobertura']*100)} de los casos con "
               f"{pct(T['A']['agarre_seguro_aceptados'])} de agarres seguros."))
    S.append(B(f"<b>C – Escasez (12 vistas/objeto):</b> {pct(T['C']['acc_clase']*100)} de clase y {pct(T['C']['pct_dentro_10'])} dentro de ±10°. "
               f"La regla de rechazo acepta solo {pct(T['C']['cobertura']*100)} de los casos ({pct(T['C']['agarre_seguro_aceptados'])} seguros entre los aceptados): "
               "con tan pocas vistas el sistema no puede certificar el ángulo de la mayoría de las piezas."))
    S.append(B(f"<b>B – Arco ciego de 60°:</b> la clase se mantiene ({pct(T['B']['acc_clase']*100)}) pero el ángulo no es recuperable por comparación de vistas "
               f"(error medio {f1(T['B']['mae_angulo'])}°, {pct(T['B']['pct_dentro_10'])} dentro de ±10°). El umbral validado acepta apenas "
               f"{pct(T['B']['cobertura']*100)} de los casos y en prueba solo {pct(T['B']['agarre_seguro_aceptados'])} de ellos son seguros (objetivo 95%); "
               "la política correcta es rechazar la estimación angular y escalar."))
    S.append(B(f"<b>D – Objetos desconocidos:</b> el sistema rechaza {pct(R['D']['test']['rechazo_desconocidos_combinado'])} de las vistas de objetos nunca vistos "
               f"(AUROC {R['D']['test']['auroc_conf']:.3f}). El costo es un falso rechazo alto en objetos conocidos durante la validación."))
    S.append(B(f"<b>Hallazgo clave:</b> una línea base de 1-NN sobre píxeles iguala o supera a HOG+SVM en varios protocolos (C: {pct(nn['C']['acc_clase']*100)} de clase). "
               "COIL-100 es un dataset de laboratorio casi resuelto por correspondencia de plantillas; el valor del SVM está en la probabilidad calibrada para decidir, no en la exactitud bruta."))

    # ---------------------------------------------------------------- 6.2 diseno
    S.append(H2("Diseño experimental y control de fuga"))
    S.append(B("<b>Pipeline:</b> descriptores (HOG 64×64 gris + histograma HSV 16/8/8, o píxeles 32×32 RGB) → StandardScaler → PCA(300) → clasificador. Algoritmos comparados: SVM-RBF, regresión logística, Random Forest y KNN (coseno)."))
    S.append(B("<b>Protocolos:</b> A (vistas cada 10°, prueba en el resto), B (arco ciego de 60° por objeto, inicio aleatorio, semilla 42), C (vistas cada 30°), D (10 objetos desconocidos, semilla 42)."))
    S.append(B("<b>Validación sin tocar la prueba:</b> pliegues con la misma geometría que el protocolo (A y C intercalados, B por sectores de 60°) y retención de objetos completos (GroupKFold) para simular desconocidos."))
    S.append(B(f"<b>Trazabilidad:</b> semilla 42, huella SHA-256 del dataset <font face='Sans-B'>{meta['sha']}</font>, métricas y parámetros registrados en MLflow."))

    # ---------------------------------------------------------------- 6.3 resultados por protocolo
    S += H1("Resultados por protocolo")
    rows = [["Métrica (prueba)", "A Interpolación", "B Arco ciego", "C Escasez", "D Desconocidos"]]
    rows.append(["Descriptores elegidos (CV)"] + [fsname[fin[p]] for p in "ABCD"])
    rows.append(["Vistas de entrenamiento / prueba"] + [f"{R[p]['n_train']:,} / {R[p]['n_test']:,}".replace(",", ".") for p in "ABCD"])
    rows.append(["Acierto de clase en validación (CV)"] + [pct(max(r["cv_acc"] for r in R[p]["comparacion_cv"]) * 100) for p in "ABCD"])
    rows.append(["Acierto de clase (prueba)"] + [pct(T[p]["acc_clase"] * 100) for p in "ABC"] + ["n/a"])
    rows.append(["Error angular medio / mediana"] + [f"{T[p]['mae_angulo']:.1f}° / {T[p]['mediana_angulo']:.1f}°" for p in "ABC"] + ["n/a"])
    rows.append(["Imágenes dentro de ±10°"] + [pct(T[p]["pct_dentro_10"]) for p in "ABC"] + ["n/a"])
    rows.append(["Agarre seguro sin rechazo (clase correcta y ±10°)"] + [pct(T[p]["agarre_seguro_sin_rechazo"]) for p in "ABC"] + ["n/a"])
    rows.append(["Cobertura con rechazo"] + [pct(T[p]["cobertura"] * 100) for p in "ABC"] + [pct(100 - R["D"]["test"]["rechazo_desconocidos_combinado"]) + " aceptados"])
    rows.append(["Agarre seguro entre aceptados"] + [pct(T[p]["agarre_seguro_aceptados"]) for p in "ABC"] + ["n/a"])
    rows.append(["Precisión@5 (recuperación)"] + [f"{T[p]['precision_at5']:.3f}" for p in "ABC"] + ["n/a"])
    rows.append(["ECE cruda → calibrada"] + [f"{T[p]['ece_cruda']:.2f} → {T[p]['ece_calibrada']:.2f}" for p in "ABC"] + ["n/a"])
    rows.append(["AUROC detección de fallo (incertidumbre)"] + [f"{T[p]['auroc_deteccion_fallo']:.2f}" for p in "ABC"] + [f"{R['D']['test']['auroc_conf']:.2f} (conf. de clase)"])
    S.append(table(rows, [150, 86, 86, 86, 96]))
    S.append(P("Los umbrales de rechazo se fijan en validación con objetivos explícitos: ≥99% de acierto de clase entre aceptados, ≥95% de agarres seguros (clase correcta y error ≤10°) y ≤5% de falsa aceptación de desconocidos simulados. "
               "«Agarre seguro sin rechazo» es lo que ocurriría si el robot actuara siempre; «entre aceptados» es lo que ocurre con el rechazo activo.", "cap"))
    S.append(H2("Lectura por protocolo"))
    pa = R["A"]["angulo_params"]
    S.append(B(f"<b>A:</b> con vistas cada 10° la estimación del ángulo es casi un problema de interpolación entre vecinas. El error medio de {T['A']['mae_angulo']:.1f}° incluye una cola pequeña: "
               f"{ea['A']['pct_error_mayor_90']:.1f}% de las imágenes con error mayor a 90° (≈{ea['A']['pct_error_cerca_180_entre_errores_gt_10']:.0f}% de los errores mayores a 10° son vistas invertidas). La regla de rechazo, calibrada con la geometría de validación (que ve la mitad de las vistas), es conservadora: "
               f"acepta {pct(T['A']['cobertura']*100)} cuando el modelo podría actuar sobre más."))
    S.append(B(f"<b>B:</b> el {ea['B']['pct_error_cerca_180']:.0f}% de las imágenes tiene un error cercano a 180° (vista opuesta) y el {ea['B']['pct_error_mayor_90']:.0f}% supera 90°: en el arco ciego "
               "no existe ninguna vista de entrenamiento cercana y la comparación de apariencia confunde el frente con el reverso. "
               f"No es un problema de hiperparámetros: la información no está en el entrenamiento (ni el 1-NN sobre píxeles pasa de {nn['B']['pct_dentro_10']:.0f}% dentro de ±10°). "
               "Recomendación operativa: usar solo la clase y pedir una segunda vista o intervención humana para el ángulo."))
    S.append(B(f"<b>C:</b> la separación de 30° deja hasta 15° entre la vista real y la más cercana, así que cumplir ±10° exige interpolar. Se logran {pct(T['C']['pct_dentro_10'])} dentro de ±10°; "
               "la validación solo con entrenamiento (retener una de cada tres de las 12 vistas) es más escasa que la prueba (hasta 30° de hueco), por lo que los umbrales resultan muy conservadores."))
    S.append(B(f"<b>D:</b> sin ver nunca estos 10 objetos, la confianza de clase los separa de los conocidos con AUROC {R['D']['test']['auroc_conf']:.2f} y la distancia al vecino más cercano con {R['D']['test']['auroc_dmin']:.2f}."))

    # ---------------------------------------------------------------- 6.4 algoritmos
    S += H1("Selección de algoritmos y descriptores")
    S.append(panel(img("models.png", W - LM - RM - 16)))
    S.append(P("SVM-RBF gana en validación en los cuatro protocolos; Random Forest queda último en los cuatro protocolos; no se investigó la causa.", "cap"))
    S.append(P("<b>Descriptores.</b> La selección entre HOG+color y píxeles 32×32 se hizo por acierto de clase en validación (solo entrenamiento) y, en empate, se prefiere el más simple. "
               "Transparencia: los píxeles se añadieron como candidato después de ver que la línea base 1-NN los favorecía en la prueba del protocolo C; el criterio de elección sigue siendo la validación, "
               "y la tabla siguiente muestra ambos conjuntos en prueba para que el lector juzgue."))
    rows = [["Protocolo", "Descriptor", "Acc. CV", "Clase (prueba)", "±10° (prueba)", "Elegido por CV"]]
    for p in "ABC":
        for fs, tag in (("hogcolor", "HOG + color"), ("pix", "Píxeles 32×32")):
            r = R[p] if fin[p] == fs else alt[p]
            rows.append([p, tag, pct(max(x["cv_acc"] for x in r["comparacion_cv"]) * 100), pct(r["test"]["acc_clase"] * 100),
                         pct(r["test"]["pct_dentro_10"]), "sí" if fin[p] == fs else ""])
    S.append(table(rows, [52, 100, 64, 90, 90, 90]))

    # ---------------------------------------------------------------- 6.5 baselines
    S.append(H2("Línea base simple y modelo de referencia moderno"))
    S.append(P("Para justificar la complejidad se comparan tres modelos con las mismas particiones: (1) 1-NN con correlación sobre píxeles 32×32 (sin entrenamiento), (2) el pipeline final SVM, y (3) una CNN pequeña "
               "(3 bloques convolucionales, 2 cabezas: clase y (sen, cos) del ángulo; TensorFlow-CPU) como referencia de aprendizaje profundo."))
    rows = [["Protocolo", "Modelo", "Clase", "Error medio", "±10°", "ms / imagen", "Ajuste"]]
    for p in "ABC":
        rows.append([p, "1-NN píxeles", pct(nn[p]["acc_clase"] * 100), f"{nn[p]['mae_angulo']:.1f}°", pct(nn[p]["pct_dentro_10"]), f"{nn[p]['ms_por_imagen']:.2f}", f"{nn[p]['t_ajuste_s']:.1f} s"])
        rows.append(["", "Pipeline final (SVM)", pct(T[p]["acc_clase"] * 100), f"{T[p]['mae_angulo']:.1f}°", pct(T[p]["pct_dentro_10"]), "—", f"{R[p]['t_total_s']:.0f} s*"])
        rows.append(["", "CNN pequeña", pct(cnn[p]["acc_clase"] * 100), f"{cnn[p]['mae_angulo']:.1f}°", pct(cnn[p]["pct_dentro_10"]), f"{cnn[p]['ms_por_imagen']:.2f}", f"{cnn[p]['t_ajuste_s']:.0f} s"])
    S.append(table(rows, [52, 108, 56, 66, 56, 62, 60]))
    S.append(P(f"*Tiempo total del protocolo: comparación de 4 algoritmos con búsqueda de hiperparámetros, OOF, calibración y umbrales (CPU de 2 núcleos). CNN: {cnn['A']['parametros']:,} parámetros, 25 épocas (A, B), 91 épocas en C: la primera corrida de C con 25 épocas colapsó (2.0% de clase) por pocos pasos de gradiente y se repitió con un presupuesto fijo de ~1.700 pasos, decisión tomada después de ver ese resultado. Sin ajuste de hiperparámetros ni aumento de datos.".replace(",", "."), "cap"))
    S.append(P("<b>Conclusión.</b> " + meta["conclusion_baseline"]))

    # ---------------------------------------------------------------- 6.6 calibracion y rechazo
    S += H1("Calibración, umbrales y decisión del robot")
    S.append(panel(img("reliability.png", W - LM - RM - 16)))
    S.append(P("Diagramas de confiabilidad en prueba. La probabilidad cruda del SVM (Platt) es sistemáticamente sub-confiada con muchas clases; la regresión isotónica ajustada solo sobre predicciones fuera de muestra del entrenamiento la corrige. "
               "En A la calibrada vale ≈1 porque todos los aciertos son correctos y el ECE degenera a 0; en B y C el ECE residual (0.02–0.09) es la cifra informativa.", "cap"))
    S.append(H2("Qué recibe el robot"))
    S.append(P("La salida de inferencia es un contrato explícito; el robot nunca recibe un ángulo sin su decisión:"))
    rows = [["Campo", "Contenido"],
            ["object_id, confidence", "Clase predicha y probabilidad calibrada (isotónica)."],
            ["angle_deg, angle_uncertainty_deg", "Ángulo estimado (media circular ponderada de las vistas de la clase) e incertidumbre = desviación circular."],
            ["nearest_view_distance", "Distancia coseno a la vista de entrenamiento más cercana (señal de objeto fuera de distribución)."],
            ["decision", "<b>GRASP</b> si confianza, incertidumbre y distancia cumplen los tres umbrales; <b>RETRY_VIEW</b> si falla solo el ángulo; <b>REJECT / HUMAN</b> si la clase es dudosa o parece objeto desconocido."],
            ["thresholds_version", "Versión de umbrales y huella del dataset con que se validaron."]]
    S.append(table(rows, [140, W - LM - RM - 140], zebra=True))
    S.append(Spacer(1, 6))
    S.append(P(f"<b>Criterio de aceptación (±10°).</b> Un agarre se considera seguro si la clase es correcta y el error angular circular ≤10°. Los umbrales se eligen para que, entre los casos aceptados, el 95% o más sean seguros "
               f"en validación. En prueba esto se cumple en A ({pct(T['A']['agarre_seguro_aceptados'])}) y queda ligeramente por debajo en C ({pct(T['C']['agarre_seguro_aceptados'])}); en B no se cumple "
               f"({pct(T['B']['agarre_seguro_aceptados'])} con una cobertura de apenas {pct(T['B']['cobertura']*100)}). Esto último es un resultado, no un defecto de reporte: el objetivo de ±10° no se puede garantizar con esos datos."))
    S.append(panel(img("risk_coverage.png", 0.5 * (W - LM - RM))))
    S.append(P("Curva riesgo–cobertura en prueba (diagnóstica, no usada para decidir umbrales): ordenar por incertidumbre eleva la tasa de agarres seguros a costa de cobertura. La estrella marca el punto operativo fijado en validación.", "cap"))

    # ---------------------------------------------------------------- 6.7 desconocidos
    S += H1("Protocolo D: objetos desconocidos")
    Rd = R["D"]; Td = Rd["test"]
    S.append(panel(img("unknown.png", W - LM - RM - 16)))
    S.append(P("Izquierda: confianza de clase en vistas conocidas (validación), desconocidos simulados con objetos retenidos del entrenamiento y los 10 desconocidos reales. Derecha: distancia al vecino más cercano.", "cap"))
    S.append(B(f"<b>Rechazo de desconocidos:</b> {pct(Td['rechazo_desconocidos_combinado'])} con la regla combinada ({pct(Td['rechazo_desconocidos_solo_clase'])} solo con la confianza de clase); "
               f"solo el {pct(Td['aceptados_desconocidos'])} de las vistas desconocidas se aceptaría como un objeto conocido."))
    S.append(B(f"<b>Costo:</b> en validación la regla rechaza el {pct(Td['falso_rechazo_conocidos_oof'])} de las vistas de objetos conocidos. El umbral se fijó para no pasar del 5% de falsa aceptación de desconocidos simulados; "
               "si la planta prefiere menos falsos rechazos, la curva ROC (AUROC "
               f"{Td['auroc_conf']:.2f}) permite mover el punto de operación explícitamente con el costo de cada error."))
    S.append(B(f"<b>Mejor señal individual:</b> la distancia a la vista más cercana separa desconocidos con AUROC {Td['auroc_dmin']:.2f} (HOG+color) y {alt['D']['test']['auroc_dmin']:.2f} (píxeles), por encima de la confianza de clase; "
               "podría darse mayor peso a esa señal en producción (a validar)."))

    # ---------------------------------------------------------------- 6.8 errores
    S.append(KeepTogether([H2("Análisis de errores"), panel(img("error_dist.png", W - LM - RM - 16)),
                           P("Distribución del error angular por protocolo y error medio según el ángulo real.", "cap")]))
    c = ea["C"]["top_confusiones"]
    S.append(B(f"<b>Confusiones de clase (C):</b> {c[0]['real']}↔{c[1]['real']} (exprimidor y cucharón de madera/crema, {c[0]['n']} y {c[1]['n']} imágenes) concentran la mayor parte del error; "
               "con 12 vistas ambos objetos se confunden entre sí (no se investigó la causa). Les siguen el portalámparas (66) y el paquete de chicle (67), ambos clasificados como la grapadora (68)."))
    def _err_stats(p):
        o = np.load(f"results/{p}{'' if fin[p] == 'hogcolor' else '_' + fin[p]}_out.npz")
        w = o["pred_obj"] != o["true_obj"]
        return int(w.sum()), int((o["conf"][w] >= R[p]["umbrales"]["confianza_clase_cruda"]).sum()), int(o["accepted"][w].sum())
    eC, eB = _err_stats("C"), _err_stats("B")
    S.append(B(f"<b>Los errores de clase son de baja confianza:</b> en C, {eC[0]} imágenes con clase errada; solo {eC[1]} superan el umbral de confianza y ninguna ({eC[2]}) es aceptada por la regla completa. "
               f"En B, {eB[0]} errores y {eB[2]} aceptados. Los errores más confiados de la galería tienen confianza cruda de 0.14–0.30."))
    S.append(B(f"<b>Ambigüedad angular por simetría:</b> en C el {ea['C']['pct_error_cerca_180_entre_errores_gt_10']:.0f}% de los errores mayores a 10° son vistas invertidas (≈180°); "
               "tazas con asa, frascos, cajas y piezas de madera casi simétricas se ven iguales de frente y de espalda. Con una sola vista, el ángulo de estos objetos es ambiguo."))
    S.append(B(f"<b>Clase errada con ángulo bueno:</b> {ea['A']['pct_fallo_clase_con_angulo_ok']:.1f}% (A), {ea['B']['pct_fallo_clase_con_angulo_ok']:.1f}% (B), {ea['C']['pct_fallo_clase_con_angulo_ok']:.1f}% (C): casi nunca ocurre, "
               "por lo que conviene decidir sobre clase y ángulo por separado (GRASP / RETRY_VIEW / REJECT)."))
    S.append(panel(img("gallery_C.png", W - LM - RM - 16)))
    S.append(P("Protocolo C. Filas 1–2: errores de clase más confiados (imagen real y un ejemplar del objeto predicho). Fila 3: peores errores angulares con clase correcta.", "cap"))
    S.append(panel(img("gallery_B.png", W - LM - RM - 16)))
    S.append(P("Protocolo B (arco ciego). Los errores de ángulo se concentran en cajas y paquetes rectangulares, casi simétricos a 180°; los de clase, en objetos que se parecen desde el costado (autos de juguete ↔ paquete de chicle).", "cap"))

    # ---------------------------------------------------------------- 6.9 model card
    S.append(PageBreak())
    S += H1("Model Card")
    rows = [["Campo", "Descripción"],
            ["Modelo", "Pipeline clásico: descriptores (HOG+HSV o píxeles 32×32) → StandardScaler → PCA(300) → SVM-RBF con probabilidades (Platt) + calibración isotónica; ángulo por posterior sobre vistas de la clase predicha (media circular); recuperación Top-K por similitud coseno."],
            ["Uso previsto", "Identificar el SKU y estimar su ángulo (0°–360°) en una estación de pick & pack, entregando al robot una decisión GRASP / RETRY_VIEW / REJECT con incertidumbre."],
            ["Datos", f"COIL-100: 7.200 imágenes, 100 objetos, 72 vistas cada 5°, fondo negro, iluminación fija. SHA-256 {meta['sha']}. Sin datos personales."],
            ["Evaluación", "Cuatro protocolos (A–D), semilla 42, validación geométrica solo con entrenamiento, una única evaluación en prueba. Tolerancia ±10°."],
            ["Resultados clave", f"A: {pct(T['A']['acc_clase']*100)} clase, {pct(T['A']['pct_dentro_10'])} ±10°. C: {pct(T['C']['acc_clase']*100)} clase, {pct(T['C']['pct_dentro_10'])} ±10°. B: ángulo no certificable. D: {pct(Td['rechazo_desconocidos_combinado'])} de desconocidos rechazados."],
            ["Limitaciones", "Dataset de laboratorio (fondo negro, sin oclusión, iluminación constante); un solo plano de rotación; objetos casi simétricos son ambiguos a 180°; las cifras no se extrapolan a una banda real sin una evaluación en planta."],
            ["Fuera de alcance", "Operar sin la regla de rechazo; garantizar ±10° con 12 vistas por objeto (C) o en arcos sin observar (B); reconocer objetos nuevos sin reentrenar; reemplazar por una CNN sin justificar costo y latencia."],
            ["Riesgos y mitigación", "Agarre fuera de ±10° → regla de rechazo triple (confianza, incertidumbre, distancia) y modo RETRY_VIEW. Deriva por iluminación/suciedad → monitoreo de distancia al vecino más cercano y del ECE; recalibración periódica."],
            ["Monitoreo", "Tasa de rechazo, distribución de la distancia al vecino más cercano, ECE con etiquetas de auditoría, tasa de reintentos y de intervención humana; alerta ante cambios sostenidos."]]
    S.append(table(rows, [90, W - LM - RM - 90]))

    S.append(H2("Limitaciones del propio estudio"))
    S.append(B("Las cifras de A y C provienen de una sola partición por protocolo (semilla 42); no se reportan intervalos de confianza entre semillas. Con 7.200 imágenes, un punto porcentual en B equivale a 12 imágenes."))
    S.append(B("Los píxeles como descriptor se incorporaron después de observar la línea base en prueba; la elección final se hizo por validación, pero el candidato lo motivó esa observación."))
    S.append(B("La CNN es una referencia de tamaño pequeño con épocas fijas, sin ajuste de hiperparámetros ni aumento de datos; no debe leerse como el mejor resultado posible del aprendizaje profundo."))
    S.append(B("Los umbrales de rechazo son conservadores porque la validación solo con entrenamiento ve una geometría más escasa que la prueba (en C y B no es posible reproducirla exactamente)."))

    S.append(H2("Reproducibilidad"))
    S.append(B("Notebook <font face='Sans-B'>Indra_MLModel_Vision.ipynb</font> (secciones 9 en adelante) y módulos <font face='Sans-B'>lib.py</font> / <font face='Sans-B'>experiment.py</font>; resultados en <font face='Sans-B'>results/</font>."))
    S.append(B("Semilla 42; mismas particiones para todos los modelos; MLflow con parámetros, métricas y huella del dataset por ejecución."))

    doc.build(S)
    return out


if __name__ == "__main__":
    print(build())
