import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Configuración ---
INPUT_DIR = "../dataset/real"
OUTPUT_DIR = "../dataset/real_augmented"
AUGMENTATIONS_PER_IMAGE = 3  # Número de imágenes augmentadas que quieres por cada imagen original

# --- Crear carpeta de salida si no existe ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Crear un generador de augmentación ---
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=10,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# --- Recorrer todas las imágenes originales ---
for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(INPUT_DIR, filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)

        # Aplicar augmentaciones
        aug_iter = datagen.flow(img, batch_size=1)
        for i in range(AUGMENTATIONS_PER_IMAGE):
            aug_img = next(aug_iter)[0].astype(np.uint8)
            save_path = os.path.join(OUTPUT_DIR, f"aug_{i}_{filename}")
            aug_img = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, aug_img)

print("✅ Augmentaciones completadas y guardadas en:", OUTPUT_DIR)
