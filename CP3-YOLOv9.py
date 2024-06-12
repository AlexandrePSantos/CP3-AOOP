from ultralytics import YOLO
import matplotlib.pyplot as plt 
import cv2
import torch

model = YOLO('yolov9c.pt')

# print(model)

# Configuração da câmera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar a detecção
    results = model(frame)

    # Processar os resultados
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result
        label = f'{model.names[int(cls)]} {conf:.2f}'
        
        # Desenhar a caixa delimitadora e o rótulo no frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Mostrar o frame com as detecções
    cv2.imshow('Detecção em Tempo Real', frame)
    
    # Pressione 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()