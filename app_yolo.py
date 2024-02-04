import streamlit as st
import cv2
import numpy as np

# Cargar el modelo YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Título de la aplicación
st.title("YOLO Object Detection")

# Subida de imágenes
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Leer la imagen
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

    # Obtener dimensiones de la imagen
    height, width, channels = image.shape

    # Preprocesar la imagen
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Obtener información de detección
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Coordenadas del cuadro delimitador
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Esquinas del cuadro delimitador
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Dibujar cuadros delimitadores y etiquetas en la imagen
    for i in range(len(boxes)):
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = (0, 255, 0)
        cv2.rectangle(image, (int(boxes[i][0]), int(boxes[i][1])), (int(boxes[i][0] + boxes[i][2]), int(boxes[i][1] + boxes[i][3])), color, 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (int(boxes[i][0]), int(boxes[i][1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Mostrar la imagen con los cuadros delimitadores
    st.image(image, channels="BGR")
