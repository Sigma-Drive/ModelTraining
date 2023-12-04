import cv2
import numpy as np

# Charger le modèle YOLO
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Charger les noms des classes
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Lire la vidéo
cap = cv2.VideoCapture('yolov5/video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir l'image en blob
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Effectuer la détection
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Traiter les résultats...

    # Afficher le résultat
    cv2.imshow('Frame', frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
