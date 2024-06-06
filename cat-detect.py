# Importación de librerías
import torch
import cv2
import numpy as np
import pandas
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath

# Lectura del modelo
print("Leyendo el modelo....")
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/David/Documents/David Contreras/UNLAM/2024/Project/Yolo/Object Detection 1/model/gatos.pt', force_reload=True, trust_repo=True)

print("Iniciando captura del streaming....")
# Video captura
cap = cv2.VideoCapture("rtsp://158.23.168.30:8554/streamig/hello")

if not cap.isOpened():
    print("Error: No se puede abrir el flujo de video")
    exit()

# Obtén el ancho y alto del video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Inicializa el VideoWriter
output_path = 'C:/Users/David/Documents/David Contreras/UNLAM/2024/Project/Yolo/Object Detection 1/screenshots/output_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec para AVI
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

print("Iniciando análisis del video.....")
# Inicializa el contador de imágenes
image_counter = 0

# Define el umbral de confianza
confidence_threshold = 0.5

while(True):
    print("Lectura de frames...")
    # Lectura de frames
    ret, frame = cap.read()
    if not ret:
        print("Error: No se puede leer el frame del video")
        break
    
    print("Detectando con el modelo...")
    # Detecciones
    detect = model(frame)
    
    # Filtrar las detecciones por probabilidad de detección
    detections = detect.pred[0]
    filtered_detections = []
    for *box, conf, cls in detections:
        if conf >= confidence_threshold:
            filtered_detections.append([*box, conf, cls])
    
    # Si hay detecciones filtradas, renderizar y guardar el frame
    if filtered_detections:
        # Crear un nuevo tensor de detecciones filtradas
        detect.pred[0] = torch.tensor(filtered_detections)
        
        print("Detección...")
        # Renderiza las detecciones
        rendered_frame = np.squeeze(detect.render())
        
        # Guarda el fotograma procesado en el archivo de video
        out.write(rendered_frame)
        
        # Genera un nombre de archivo único usando el contador
        file_path = f'C:/Users/David/Documents/David Contreras/UNLAM/2024/Project/Yolo/Object Detection 1/screenshots/ss_{image_counter:04d}.jpeg'
        cv2.imwrite(file_path, rendered_frame)
        
        # Mostramos los FPS
        cv2.imshow('Detector de Gatos', rendered_frame)

        image_counter += 1

    # Lee el teclado, para terminar presionamos la tecla Escape
    if cv2.waitKey(5) == 27: 
        break

# Finalizamos el programa liberando los recursos
cap.release()
out.release()
cv2.destroyAllWindows()