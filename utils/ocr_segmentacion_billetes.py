import io
import os
import cv2
import uuid
import numpy as np
from collections import defaultdict
from google.cloud import vision

client = vision.ImageAnnotatorClient()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials/vision-service.json"

def segmentar_billetes_por_texto(imagen_path: str, guardar_debug=False) -> list:
    with io.open(imagen_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    annotations = response.text_annotations

    if not annotations:
        return []

    img = cv2.imread(imagen_path)
    img_height, img_width = img.shape[:2]

    zonas_por_fila = defaultdict(list)

    for annotation in annotations[1:]:
        vertices = annotation.bounding_poly.vertices
        x = min(v.x for v in vertices)
        y = min(v.y for v in vertices)
        x_max = max(v.x for v in vertices)
        y_max = max(v.y for v in vertices)
        h = y_max - y

        # agrupación por fila (eje y)
        fila = int(y // 100)
        zonas_por_fila[fila].append((x, y, x_max, y_max))

    rutas_billetes = []
    for fila, zonas in zonas_por_fila.items():
        if not zonas:
            continue
        x1 = min(z[0] for z in zonas)
        y1 = min(z[1] for z in zonas)
        x2 = max(z[2] for z in zonas)
        y2 = max(z[3] for z in zonas)

        # márgenes
        margin = 20
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(img_width, x2 + margin)
        y2 = min(img_height, y2 + margin)

        billete = img[y1:y2, x1:x2]
        nombre_archivo = f"temp_images/billete_texto_{uuid.uuid4().hex[:8]}.jpg"
        cv2.imwrite(nombre_archivo, billete)
        rutas_billetes.append(nombre_archivo)

        if guardar_debug:
            os.makedirs("debug", exist_ok=True)
            cv2.imwrite(f"debug/billete_debug_{uuid.uuid4().hex[:8]}.jpg", billete)

    return rutas_billetes
