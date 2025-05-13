import os

# Ruta de la carpeta donde están las imágenes que quieres renombrar
folder_path = r"C:\Users\Alsan\OneDrive\Imágenes\Dataset\reales\archive\USA currency\100 Dollar"

# Prefijo que quieres para el nombre de las imágenes
prefix = "100dollar"

# Contador inicial
counter = 1

# Recorrer todos los archivos de la carpeta
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Construir el nuevo nombre
        new_name = f"{prefix}_{counter}.jpg"  # Vamos a convertir todo a .jpg para uniformidad
        # Rutas completa: original y destino
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_name)
        # Renombrar archivo
        os.rename(src, dst)
        counter += 1

print("✅ Imágenes renombradas correctamente.")
