import cv2
import numpy as np
import os

def centrar_y_recortar_rostros(imagen_path, tamano_recorte=(200, 200)):
    # Cargar la imagen
    imagen = cv2.imread(imagen_path)
    if imagen is None:
        print(f"No se pudo cargar la imagen: {imagen_path}")
        return []
    
    # Convertir la imagen a escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Cargar el clasificador de rostros pre-entrenado
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detectar rostros en la imagen
    rostros = face_cascade.detectMultiScale(gris, 1.3, 5)
    
    rostros_recortados = []
    
    for (x, y, w, h) in rostros:
        # Calcular el centro del rostro
        centro_x, centro_y = x + w // 2, y + h // 2
        
        # Calcular las coordenadas para el recorte
        inicio_x = max(0, centro_x - tamano_recorte[0] // 2)
        inicio_y = max(0, centro_y - tamano_recorte[1] // 2)
        fin_x = min(imagen.shape[1], inicio_x + tamano_recorte[0])
        fin_y = min(imagen.shape[0], inicio_y + tamano_recorte[1])
        
        # Recortar el rostro
        rostro_recortado = imagen[inicio_y:fin_y, inicio_x:fin_x]
        
        # Redimensionar si es necesario
        if rostro_recortado.shape[:2] != tamano_recorte:
            rostro_recortado = cv2.resize(rostro_recortado, tamano_recorte)
        
        rostros_recortados.append(rostro_recortado)
    
    return rostros_recortados

# Lista de rutas de imágenes
imagenes_path = [
    'IMG_20241014_125616.jpg',
    'IMG_20241014_125749.jpg',
    'IMG_20241014_125914.jpg',
    'IMG_20241014_130012.jpg',
    'IMG_20241014_130033.jpg',
    'IMG_20241014_130154.jpg',
    'IMG_20241014_130221.jpg',
    'IMG_20241014_130249.jpg',
    'IMG_20241014_130311.jpg',
    'IMG_20241014_130340.jpg'
    # Agrega aquí más rutas de imágenes según sea necesario
]

tamano_recorte = (1500, 1500)  # Tamaño deseado para los recortes

# Directorio para guardar todos los rostros recortados
directorio_salida = 'Resultado'
os.makedirs(directorio_salida, exist_ok=True)

total_rostros = 0

for imagen_path in imagenes_path:
    if os.path.exists(imagen_path):
        # Obtener el nombre del archivo sin la extensión
        nombre_base = os.path.splitext(os.path.basename(imagen_path))[0]
        
        rostros_recortados = centrar_y_recortar_rostros(imagen_path, tamano_recorte)
        
        if rostros_recortados:
            # Guardar los rostros recortados
            for j, rostro in enumerate(rostros_recortados, 1):
                nombre_archivo = f'{nombre_base}_rostro_{j}.jpg'
                ruta_completa = os.path.join(directorio_salida, nombre_archivo)
                cv2.imwrite(ruta_completa, rostro)
        
            total_rostros += len(rostros_recortados)
            print(f"Se han recortado {len(rostros_recortados)} rostros de la imagen {nombre_base}.")
        else:
            print(f"No se detectaron rostros en la imagen {nombre_base}.")
    else:
        print(f"La imagen no existe en la ruta especificada: {imagen_path}")

print(f"Total de rostros recortados: {total_rostros}")
print(f"Los rostros recortados se han guardado en el directorio: {directorio_salida}")