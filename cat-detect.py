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
print("Iniciando analisis del video.....")
# analisis del video 

while(True):
    print("lectura de frames..")
    # Lectura de frames 
    ret, frame = cap.read()
    print("Detectando con el modelo...")
    # detecciones 
    detect = model(frame)
    print("deteción..")

    # obtenemos info de las imagens detectadas

    # Mostramos los FPS
    cv2.imshow('Detector de Gatos', np.squeeze(detect.render()))
    cv2.imwrite('C:/Users/David/Documents/David Contreras/UNLAM/2024/Project/Yolo/Object Detection 1/screenshots/ss.jpeg', np.squeeze(detect.render()) )

    #lee el teclad, para terminar presionamos la tecla scape
    t = cv2.waitKey(5)
    if t == 27: 
        break

# Finalizamos el programa  liberando los recursos
cap.release()
cv2.destroyAllWindows()
