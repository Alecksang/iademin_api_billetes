# validacion_visual.py
import os
import cv2
import numpy as np
from google.cloud import vision

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials/vision-service.json"
client = vision.ImageAnnotatorClient()

PALABRAS_INVALIDAS = [
    "PACIFIC", "POWER", "UGOKLIN", "BREKY", "DEMATITE", "LEVAL", "FENDER", "MOTE",
    "LOTE", "MOUN", "BASE", "LATES", "CRAMERICA", "OPAMERICAN", "PRNATE", "PUBRIC", "DOLLRS",
    "AMERIGA", "STATESOF", "UNITED'S", "GEM", "LANG", "PAWER", "BERES", "MAWAKIN", "TENDUM",
    "REPÚBLICA", "BOLIVARIANA", "VENEZUELA", "BOLIVARES", "PESOS", "EUROS", "EURO", "BOLIVAR"
]

def es_billete_visualmente_valido(ruta_imagen: str, debug: bool = False) -> bool:
    try:
        img = cv2.imread(ruta_imagen)
        if img is None:
            if debug: print("Imagen no cargada")
            return False

        # Contraste
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contraste = gris.std()
        if debug: print(f"Contraste: {contraste}")
        if contraste < 27:
            if debug: print("Falla: bajo contraste")
            return False

        # Saturación
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturacion_prom = hsv[:, :, 1].mean()
        if debug: print(f"Saturación promedio: {saturacion_prom}")
        if saturacion_prom < 16:
            if debug: print("Falla: baja saturación")
            return False

        # Texto sospechoso
        with open(ruta_imagen, "rb") as image_file:
            content = image_file.read()
        vision_image = vision.Image(content=content)
        response = client.text_detection(image=vision_image)
        texts = response.text_annotations

        if texts:
            texto_completo = texts[0].description.upper()
            coincidencias = [p for p in PALABRAS_INVALIDAS if p in texto_completo]
            if debug: print(f"Palabras sospechosas detectadas: {coincidencias}")
            if len(coincidencias) >= 2:
                if debug: print("Falla: múltiples palabras sospechosas encontradas")
                return False

        return True

    except Exception as e:
        if debug: print(f"Error visual: {e}")
        return False
