# importación de librerías
import torch
import cv2
import numpy as np 
import pandas 
import pathlib
import keyboard
import time
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Usando dispositivo: {device}')
# lectura del modelo
print("leyendo el modelo....")
model = torch.hub.load('ultralytics/yolov5','custom',path='C:/Users/User/Desktop/autodetec/crack_v1.pt', force_reload=True, trust_repo=True).to(device)
#model = torch.hub.load('.','custom',path = 'C:/Users/David/Documents/David Contreras/UNLAM/2024/Project/Yolo/Object Detection 1/model/gatos.pt',source='local')

print("Iniciando captura del streaming....")
# Video captura 
#cap = cv2.VideoCapture(0)
cap= None

while not cap or not cap.isOpened():
    if keyboard.is_pressed('esc'):
        print("Tecla 'Esc' presionada. Deteniendo el script.")
        exit()
    try:
        cap = cv2.VideoCapture("rtmp://localhost:1935/live")
    except :
        print("se capturo el error")
        cap=None
    print("aprete esc para detener el script")
    time.sleep(1)
    print("ya paso el tiempo")   


# Obtén el ancho y alto del video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Inicializa el VideoWriter
output_path = 'C:/Users/User/Desktop/autodetec/demo/video/output_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec para AVI
out = cv2.VideoWriter(output_path, fourcc, 20, (frame_width, frame_height))

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
    print("detección..")

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

    # Lee el teclado, para terminar presionamos la tecla Escape
    if cv2.waitKey(5) == 27: 
        break

# Finalizamos el programa  liberando los recursos
cap.release()
out.release()
cv2.destroyAllWindows()