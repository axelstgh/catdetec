# importación de librerías
import torch
import cv2
import numpy as np 
import pandas 
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath


# lectura del modelo
print("leyendo el modelo....")
model = torch.hub.load('ultralytics/yolov5','custom',path='C:/Users/David/Documents/David Contreras/UNLAM/2024/Project/Yolo/Object Detection 1/model/gatos.pt', force_reload=True, trust_repo=True)
#model = torch.hub.load('.','custom',path = 'C:/Users/David/Documents/David Contreras/UNLAM/2024/Project/Yolo/Object Detection 1/model/gatos.pt',source='local')

print("Iniciando captura del streaming....")
# Video captura 
cap = cv2.VideoCapture("rtsp://158.23.168.30:8554/streamig/hello")
#cap = cv2.VideoCapture(0)

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

while(True):
    print("lectura de frames..")
    # Lectura de frames 
    ret, frame = cap.read()
    print("Detectando con el modelo...")
    # detecciones 
    detect = model(frame)
    print("deteción..")

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

# Finalizamos el programa  liberando los recursos
cap.release()
out.release()
cv2.destroyAllWindows()
