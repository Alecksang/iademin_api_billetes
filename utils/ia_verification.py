import numpy as np
import tensorflow as tf
import cv2

# Cargar el modelo entrenado (ajusta el path si está en otra carpeta)
modelo = tf.keras.models.load_model("modelo_billetes.h5")

def calcular_porcentaje_veracidad(imagen_path: str) -> float:
    """
    Usa el modelo IA para predecir el porcentaje de veracidad de un billete.

    Args:
        imagen_path (str): Ruta del archivo de imagen.

    Returns:
        float: Porcentaje entre 0 y 1 (mayor es más real).
    """
    img = cv2.imread(imagen_path)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {imagen_path}")

    # Redimensionar a la entrada esperada por el modelo
    img_resized = cv2.resize(img, (224, 224))  # Ajusta tamaño si tu modelo espera otro
    img_array = img_resized.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predecir
    resultado = modelo.predict(img_array)[0][0]
    return round(float(resultado), 4)
