import cv2
from typing import List, Dict
from processing.metrics_extractor import calcular_confianza_y_match, detectar_motivo_falsedad
from utils.bbox import remove_duplicate_billetes


def detectar_billetes_en_imagen(nombre_imagen: str, ruta_imagen: str) -> Dict:
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        return {
            "imagen": nombre_imagen,
            "error": "No se pudo cargar la imagen."
        }

    # Paso 1: Detectar regiones de billetes (mock por ahora)
    billetes_crudos = detectar_regiones_billetes(imagen)

    # Paso 2: Eliminar duplicados por superposición (IoU)
    billetes_filtrados = remove_duplicate_billetes(billetes_crudos)

    resultados = []
    for i, billete in enumerate(billetes_filtrados, start=1):
        confianza, match = calcular_confianza_y_match(billete, imagen)

        es_real = bool(match >= 0.85)
        motivo = None if es_real else detectar_motivo_falsedad(match)

        resultados.append({
            "numero": i,
            "es_real": es_real,
            "confianza": round(confianza * 100, 2),
            "match_con_modelo": round(match, 3),
            "bounding_box": billete['bbox'],
            "motivo_probable_falsedad": motivo
        })

    metricas = calcular_metricas(resultados)

    return {
        "imagen": nombre_imagen,
        "cantidad_detectada": len(resultados),
        "billetes": resultados,
        "metricas": metricas
    }


def detectar_regiones_billetes(imagen) -> List[Dict]:
    # Simulación para test inicial, reemplazar con lógica real de detección
    return [
        {"bbox": (50, 50, 300, 100)},
        {"bbox": (60, 55, 300, 100)},
        {"bbox": (400, 100, 300, 100)},
        {"bbox": (800, 150, 300, 100)},
        {"bbox": (1200, 180, 300, 100)},
        {"bbox": (60, 55, 300, 100)}  # duplicado simulado
    ]


def calcular_metricas(billetes: List[Dict]):
    total = len(billetes)
    reales = sum(1 for b in billetes if b['es_real'])
    falsos = total - reales
    confianza_prom = round(sum(b['confianza'] for b in billetes) / total, 2) if total else 0

    return {
        "billetes_real": reales,
        "billetes_falsos": falsos,
        "confianza_promedio": confianza_prom
    }
