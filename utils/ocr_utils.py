# utils/ocr_utils.py
import os
import io
import re
from typing import List
from google.cloud import vision
from PIL import Image
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials/vision-service.json"


def extraer_serial(ruta_imagen: str, debug: bool = True) -> List[str]:
    """Extrae posibles seriales de billetes desde una imagen usando Google Vision."""
    client = vision.ImageAnnotatorClient()

    with io.open(ruta_imagen, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if debug:
        print("Texto detectado por Google Vision:")
        for text in texts:
            print(text.description)

    # Definir regex para seriales válidos
    regex_serial = r"[A-Z]{1,2}\s?\d{8}\s?[A-Z]"
    seriales = []

    for text in texts:
        matches = re.findall(regex_serial, text.description.replace("\n", " "))
        for match in matches:
            serial_limpio = match.replace(" ", "").strip()
            if serial_limpio not in seriales:
                seriales.append(serial_limpio)

    return seriales
