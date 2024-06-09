# Importación de librerías
import torch
import cv2
import numpy as np
import pandas
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Usando dispositivo: {device}')
# Lectura del modelo
print("Leyendo el modelo....")
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/User/Desktop/autodetec/crack_v1.pt', force_reload=True, trust_repo=True).to(device)

print("Iniciando captura del streaming....")
# Video captura
cap = cv2.VideoCapture("rtmp://192.168.0.8:1935/live")

if not cap.isOpened():
    print("Error: No se puede abrir el flujo de video")
    exit()

# Obtén el ancho y alto del video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Inicializa el VideoWriter
output_path = 'C:/Users/User/Desktop/autodetec/demo/video/output_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec para AVI
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

print("Iniciando análisis del video.....")
# Inicializa el contador de imágenes
image_counter = 0

# Define el umbral de confianza
confidence_threshold = 0.8

# Define una variable para almacenar las detecciones anteriores
previous_detections = []
stability_counter = 0
stability_threshold = 5  # Número de frames para considerar una detección estable

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
    
    # Convertir a lista para comparación
    filtered_detections_list = [d[:4] for d in filtered_detections]  # Solo coordenadas de caja para comparación

    if filtered_detections_list == previous_detections:
        stability_counter += 1
    else:
        stability_counter = 0

    # Si los detections son estables durante el umbral de estabilidad
    if stability_counter >= stability_threshold:
        if filtered_detections:
            # Crear un nuevo tensor de detecciones filtradas
            detect.pred[0] = torch.tensor(filtered_detections)
            
            print("Detección...")
            # Renderiza las detecciones
            rendered_frame = np.squeeze(detect.render())
            
            # Guarda el fotograma procesado en el archivo de video
            out.write(rendered_frame)
            
            # Genera un nombre de archivo único usando el contador
            file_path = f'C:/Users/User/Desktop/autodetec/demo/imagenes/ss_{image_counter:04d}.jpeg'
            cv2.imwrite(file_path, rendered_frame)
            
            # Mostramos los FPS
            cv2.imshow('Detector de Gatos', rendered_frame)

            image_counter += 1

        # Reiniciar el contador de estabilidad
        stability_counter = 0

    # Actualizar las detecciones anteriores
    previous_detections = filtered_detections_list

    # Lee el teclado, para terminar presionamos la tecla Escape
    if cv2.waitKey(5) == 27: 
        break

# Finalizamos el programa liberando los recursos
cap.release()
out.release()
cv2.destroyAllWindows()