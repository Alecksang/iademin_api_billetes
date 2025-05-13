# main.py
import os
import uuid
from typing import List
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from utils.ocr_utils import extraer_serial
from utils.ia_verification import calcular_porcentaje_veracidad
from utils.validacion_visual import es_billete_visualmente_valido
from utils.ocr_segmentacion_billetes import segmentar_billetes_por_texto

# Configura las credenciales de Google Cloud Vision
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials/vision-service.json"

app = FastAPI()
THRESHOLD_REAL = 0.85  # Umbral mínimo para considerar válido

@app.post("/validar-billetes")
async def validar_billetes(files: List[UploadFile] = File(...)):
    resultados = []

    for file in files:
        original_temp_path = f"temp_images/original_{uuid.uuid4().hex[:8]}_{file.filename}"
        try:
            with open(original_temp_path, "wb") as f:
                f.write(await file.read())

            # ✅ Segmentación OCR basada en texto (agrupación por billetes)
            billetes_rutas = segmentar_billetes_por_texto(original_temp_path, guardar_debug=True)

            if not billetes_rutas:
                billetes_rutas = [original_temp_path]

            seriales_detectados = set()

            for idx, ruta_billete in enumerate(billetes_rutas):
                try:
                    posibles_seriales = extraer_serial(ruta_billete)
                    serial = posibles_seriales[0] if posibles_seriales else None
                    if not serial:
                        continue  # ⛔️ Ignorar subimagen sin serial
                    if serial in seriales_detectados:
                        continue
                    seriales_detectados.add(serial)

                    score_modelo = calcular_porcentaje_veracidad(ruta_billete)
                    visual_valido = es_billete_visualmente_valido(ruta_billete, debug=True)  # Activar debug

                    motivo = None
                    if not visual_valido:
                        porcentaje = round(score_modelo * 0.4, 4)
                        motivo = "Falla en validación visual (contraste, color, rostro o texto inválido)"
                    else:
                        porcentaje = round(score_modelo, 4)

                    es_valido = porcentaje >= THRESHOLD_REAL

                    resultado = {
                        "imagen_original": file.filename,
                        "subimagen_id": idx + 1,
                        "serial": serial,
                        "porcentaje_validacion": porcentaje,
                        "es_valido": es_valido
                    }

                    if not es_valido and motivo:
                        resultado["motivo"] = motivo

                    resultados.append(resultado)

                except Exception as e:
                    resultados.append({
                        "imagen_original": file.filename,
                        "subimagen_id": idx + 1,
                        "error": str(e)
                    })

                finally:
                    if os.path.exists(ruta_billete) and ruta_billete != original_temp_path:
                        os.remove(ruta_billete)

        except Exception as e:
            resultados.append({
                "imagen_original": file.filename,
                "error": str(e)
            })

        finally:
            if os.path.exists(original_temp_path):
                os.remove(original_temp_path)

    return resultados
