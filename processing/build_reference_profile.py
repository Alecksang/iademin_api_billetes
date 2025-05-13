import os
import json
from metrics_extractor import extraer_metricas

# --- Configuración ---
# Ruta a las imágenes reales
IMAGES_DIR = "../dataset/real/"
# Ruta donde guardarás el perfil de referencia
OUTPUT_PROFILE = "../processing/reference_profile.json"

# --- Recolectar métricas ---
metricas_totales = {
    "nitidez": [],
    "cantidad_bordes": [],
    "mean_r": [],
    "mean_g": [],
    "mean_b": [],
    "textura": [],
    "brillo": [],
    "aspect_ratio": []
}

print("🔍 Procesando imágenes reales para construir perfil de referencia...")

for filename in os.listdir(IMAGES_DIR):
    path = os.path.join(IMAGES_DIR, filename)
    if path.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            metricas = extraer_metricas(path)
            for key in metricas_totales.keys():
                metricas_totales[key].append(metricas[key])
        except Exception as e:
            print(f"⚠️ Error procesando {filename}: {e}")

# --- Calcular promedios y desvíos estándar ---
perfil_referencia = {}

for key, valores in metricas_totales.items():
    if valores:
        perfil_referencia[key] = {
            "mean": float(sum(valores) / len(valores)),
            "std": float((sum((x - sum(valores)/len(valores))**2 for x in valores) / len(valores))**0.5)
        }
    else:
        perfil_referencia[key] = {
            "mean": 0.0,
            "std": 0.0
        }

# --- Guardar perfil en JSON ---
with open(OUTPUT_PROFILE, 'w') as f:
    json.dump(perfil_referencia, f, indent=4)

print(f"✅ Perfil de referencia guardado exitosamente en: {OUTPUT_PROFILE}")
