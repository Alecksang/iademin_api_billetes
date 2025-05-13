import cv2
import os
import uuid

def rects_similares(r1, r2, umbral=0.25):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    dw = abs(w1 - w2)
    dh = abs(h1 - h2)
    return dx < w1 * umbral and dy < h1 * umbral and dw < w1 * umbral and dh < h1 * umbral

def segmentar_billetes_en_imagen(imagen_path: str, guardar_debug=False) -> list:
    img = cv2.imread(imagen_path)
    if img is None:
        return []

    altura, ancho = img.shape[:2]
    area_total = altura * ancho

    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    edges = cv2.Canny(blur, 20, 120)

    # Mejorar los bordes para cerrar contornos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilatada = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilatada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rutas_billetes = []
    rects_guardados = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 0.03 * area_total:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspecto = w / h
        if aspecto < 1.5 or aspecto > 4.0:
            continue

        rect_actual = (x, y, w, h)
        if any(rects_similares(rect_actual, r) for r in rects_guardados):
            continue

        billete_crop = img[y:y+h, x:x+w]
        nombre_archivo = f"temp_images/billete_{uuid.uuid4().hex[:8]}.jpg"
        cv2.imwrite(nombre_archivo, billete_crop)
        rutas_billetes.append(nombre_archivo)
        rects_guardados.append(rect_actual)

        if guardar_debug:
            os.makedirs("debug", exist_ok=True)
            cv2.imwrite(f"debug/billete_debug_{uuid.uuid4().hex[:8]}.jpg", billete_crop)

    return rutas_billetes
