from ultralytics import YOLO
import cv2
import numpy as np
import pygame
import threading

# Definir alturas médias para cada classe de animal (em metros)
KNOWN_HEIGHTS = {
    16: 0.5,  # Dog
    17: 0.3,  # Cat
    18: 1.6,  # Horse
    19: 1.0,  # Sheep
    20: 1.4,  # Cow
    21: 3.0,  # Elephant
    22: 2.8,  # Bear
    23: 1.5,  # Zebra
    24: 5.5,  # Giraffe
}

# Distância focal em pixeis
FOCAL_LENGTH = 700  

# Classes de animais no dataset COCO
ANIMAL_CLASSES = list(KNOWN_HEIGHTS.keys())

# Função para calcular a interseção sobre a união (IoU) de duas caixas delimitadoras
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1[:4]
    x1g, y1g, x2g, y2g = box2[:4]

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou

# Função para agrupar caixas delimitadoras próximas
def group_boxes(boxes, iou_threshold=0.5):
    if not boxes:
        return []
    
    # Ordenar por confiança
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)  
    grouped_boxes = []
    
    while boxes:
        current_box = boxes.pop(0)
        group = [current_box]
        
        boxes_copy = boxes[:]
        for other_box in boxes_copy:
            if calculate_iou(current_box, other_box) > iou_threshold:
                group.append(other_box)
                boxes.remove(other_box)
        
        grouped_boxes.append(group)
    
    return grouped_boxes

# Função para mostrar um sinal de aviso
def show_warning(frame):
    frame_height, frame_width = frame.shape[:2]
    thickness = 5
    color = (0, 0, 255)  

    cv2.line(frame, (0, 0), (frame_width, 0), color, thickness)
    cv2.line(frame, (0, frame_height - 1), (frame_width, frame_height - 1), color, thickness)
    cv2.line(frame, (0, 0), (0, frame_height), color, thickness)
    cv2.line(frame, (frame_width - 1, 0), (frame_width - 1, frame_height), color, thickness)

# Função para reproduzir som em uma thread separada
def play_sound():
    pygame.mixer.init()
    pygame.mixer.music.load('warning_sound.mp3')
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

# Carregar o modelo treinado
model = YOLO('yolov9c.pt')

# Webcam
cap = cv2.VideoCapture(0)
desired_fps = 30 
cap.set(cv2.CAP_PROP_FPS, desired_fps)

# Verificar se a vídeo foi aberto corretamente
if not cap.isOpened():
    print("Erro ao abrir a vídeo.")
    exit()

# Abrir o vídeo original para obter as informações do vídeo
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Definir o nome do arquivo de saída para o vídeo
output_video_path = 'detected_video.mp4'

# Criar o objeto VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Loop através das frames do vídeo
while True:
    # Ler a próxima frame
    ret, frame = cap.read()

    # Verificar se a frame foi lida corretamente
    if not ret:
        break
    
    # Realizar a detecção
    results = model(frame)

    # Agrupar caixas delimitadoras e filtrar por classes de animais
    boxes = []
    for result in results[0].boxes:
        x1, y1, x2, y2 = result.xyxy[0].cpu().numpy()
        conf = result.conf[0].cpu().numpy()
        cls = result.cls[0].cpu().numpy()
        
        if int(cls) in ANIMAL_CLASSES:
            boxes.append((x1, y1, x2, y2, conf, cls))

    # Agrupar caixas delimitadoras próximas
    grouped_boxes = group_boxes(boxes)

    # Flag para verificar se um objeto foi detectado
    object_detected = False

    # Desenhar caixas delimitadoras agrupadas e estimar distância
    for group in grouped_boxes:
        # Calcular a caixa delimitadora média do grupo
        x1_avg = np.mean([box[0] for box in group])
        y1_avg = np.mean([box[1] for box in group])
        x2_avg = np.mean([box[2] for box in group])
        y2_avg = np.mean([box[3] for box in group])
        cls_avg = group[0][5]  # Assumir que todas as caixas do grupo são da mesma classe

        label = f'{model.names[int(cls_avg)]}'

        # Calcular altura média do objeto em pixeis
        object_height_in_pixels = y2_avg - y1_avg
        
        # Usar a altura conhecida da classe para estimar a distância
        known_height = KNOWN_HEIGHTS[int(cls_avg)]
        if object_height_in_pixels > 0:
            distance = (known_height * FOCAL_LENGTH) / object_height_in_pixels
            distance_label = f'Distance: {distance:.2f}m'
            
            # Se a distância for menor ou igual a 10 metros, marcar que um objeto foi detectado
            if distance <= 10.0:
                object_detected = True
            else:
                color = (255, 0, 0)  

            # Desenhar a caixa delimitadora média
            cv2.rectangle(frame, (int(x1_avg), int(y1_avg)), (int(x2_avg), int(y2_avg)), color, 1)
            cv2.putText(frame, label, (int(x1_avg), int(y1_avg) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 1) 
            cv2.putText(frame, distance_label, (int(x1_avg), int(y2_avg) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 1)

    # Mostrar a frame processada
    cv2.imshow('Detecção de Animais', frame)

    # Adicionar a frame processada ao vídeo de saída
    out.write(frame)

    # Verificar se a tecla 'q' foi pressionada para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Se um objeto foi detectado, reproduzir o som de aviso
    if object_detected:
        color = (0, 0, 255) 
        show_warning(frame)
        # Reproduzir som numa thread separada
        threading.Thread(target=play_sound).start()
    else:
        color = (255, 0, 0)

# liberar os recursos
cap.release()
out.release()
cv2.destroyAllWindows()
