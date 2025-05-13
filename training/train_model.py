import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
import os

# --- Parámetros de configuración ---
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
DATASET_DIR = "../dataset"
OUTPUT_MODEL = "../modelo_billetes.h5"

# --- Data Augmentation para "falsos simulados" ---
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # Separar 20% para validación
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.5, 1.5],
    shear_range=0.2,
    fill_mode='nearest',
    horizontal_flip=True
)

# --- Generadores para entrenamiento y validación ---
train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# --- Cargar MobileNetV2 sin capas finales ---
base_model = MobileNetV2(
    weights='imagenet', 
    include_top=False, 
    input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3))
)
base_model.trainable = False  # No actualizamos pesos del modelo base

# --- Capas personalizadas arriba de MobileNetV2 ---
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)  # 1 neurona para clasificar real (0) o falso (1)

model = Model(inputs=base_model.input, outputs=output)

# --- Compilación ---
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# --- Entrenamiento ---
model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# --- Guardar modelo final ---
model.save(OUTPUT_MODEL)
print(f"✅ Modelo guardado exitosamente en: {OUTPUT_MODEL}")
