from utils.ocr_utils import extraer_serial

ruta_imagen = "temp_images/Imagen de WhatsApp 2025-04-06 a las 17.19.06_eaf1a06a.jpg"
serial = extraer_serial(ruta_imagen)
print("Serial detectado:", serial if serial else "No se detectó ningún serial.")