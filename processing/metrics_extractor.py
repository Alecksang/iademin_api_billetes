import cv2
import numpy as np
import json
from skimage.feature import local_binary_pattern
import os

# --- Parámetros para LBP
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = 'uniform'

# --- Ruta del perfil de referencia
REFERENCE_PROFILE_PATH = os.path.join("processing", "reference_profile.json")


def calcular_nitidez(img_gray):
    return cv2.Laplacian(img_gray, cv2.CV_64F).var()


def calcular_bordes(img_gray):
    edges = cv2.Canny(img_gray, threshold1=50, threshold2=150)
    return np.sum(edges > 0)


def calcular_histograma_color(img_rgb):
    chans = cv2.split(img_rgb)
    features = []
    for chan in chans:
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.append(np.mean(hist))
    return features


def calcular_textura(img_gray):
    lbp = local_binary_pattern(img_gray, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)
    (hist, _) = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, LBP_POINTS + 3),
        range=(0, LBP_POINTS + 2)
    )
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return np.mean(hist)


def calcular_brillo(img_gray):
    return np.mean(img_gray)


def calcular_aspect_ratio(img_rgb):
    h, w = img_rgb.shape[:2]
    return w / h


def cargar_perfil_referencia():
    with open(REFERENCE_PROFILE_PATH, "r") as f:
        return json.load(f)


def calcular_confianza_y_match(billete, imagen):
    x, y, w, h = billete['bbox']
    recorte = imagen[y:y+h, x:x+w]

    try:
        img_rgb = cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    except:
        return 0.0, 0.0

    # Extraer métricas del billete
    metrics = {
        "nitidez": calcular_nitidez(img_gray),
        "cantidad_bordes": calcular_bordes(img_gray),  # este nombre debe coincidir
        "mean_r": calcular_histograma_color(img_rgb)[0],
        "mean_g": calcular_histograma_color(img_rgb)[1],
        "mean_b": calcular_histograma_color(img_rgb)[2],
        "textura": calcular_textura(img_gray),
        "brillo": calcular_brillo(img_gray),
        "aspect_ratio": calcular_aspect_ratio(img_rgb),
    }

    referencia = cargar_perfil_referencia()

    diferencias = []
    for key in metrics:
        ref_mean = referencia.get(key, {}).get("mean", 1.0)
        if ref_mean == 0:
            continue
        val = metrics[key]
        dif = min(1.0, abs(val - ref_mean) / ref_mean)
        diferencias.append(dif)

    promedio_error = sum(diferencias) / len(diferencias) if diferencias else 1.0
    match_score = max(0.0, 1.0 - promedio_error)

    return round(match_score, 3), round(match_score, 3)


def detectar_motivo_falsedad(match: float) -> str:
    if match < 0.75:
        return "Baja coincidencia visual con billete real"
    elif match < 0.85:
        return "Coincidencia visual marginal"
    else:
        return "Criterio de falsedad no especificado"
