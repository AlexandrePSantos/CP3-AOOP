from ultralytics import YOLO
import cv2
import numpy as np

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

FOCAL_LENGTH = 700  # Exemplo de distância focal em pixels (necessita calibração)

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
    
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)  # Ordenar por confiança
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

# Função para mostrar um sinal de aviso na tela
def show_warning(frame):
    frame_height, frame_width = frame.shape[:2]
    thickness = 5
    color = (0, 0, 255)  # Vermelho

    # Desenhar a linha superior
    cv2.line(frame, (0, 0), (frame_width, 0), color, thickness)
    # Desenhar a linha inferior
    cv2.line(frame, (0, frame_height - 1), (frame_width, frame_height - 1), color, thickness)
    # Desenhar a linha esquerda
    cv2.line(frame, (0, 0), (0, frame_height), color, thickness)
    # Desenhar a linha direita
    cv2.line(frame, (frame_width - 1, 0), (frame_width - 1, frame_height), color, thickness)

# Carregar o modelo treinado
model = YOLO('yolov9c.pt')

# Abrir o vídeo
video_path = './cows_crossing.mp4'
cap = cv2.VideoCapture(video_path)

# Verificar se o vídeo foi aberto corretamente
if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

# Abrir o vídeo original para obter as informações do vídeo
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Definir o nome do arquivo de saída para o vídeo com detecção
output_video_path = 'detected_video.mp4'

# Criar o objeto VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Loop através dos quadros do vídeo
while True:
    # Ler o próximo quadro
    ret, frame = cap.read()

    # Verificar se o quadro foi lido corretamente
    if not ret:
        break

    # Realizar a detecção
    results = model(frame)

    # Coletar caixas delimitadoras e filtrar por classes de animais
    boxes = []
    for result in results[0].boxes:
        x1, y1, x2, y2 = result.xyxy[0].cpu().numpy()
        conf = result.conf[0].cpu().numpy()
        cls = result.cls[0].cpu().numpy()
        
        if int(cls) in ANIMAL_CLASSES:
            boxes.append((x1, y1, x2, y2, conf, cls))

    # Agrupar caixas delimitadoras próximas
    grouped_boxes = group_boxes(boxes)

    # Desenhar caixas delimitadoras agrupadas e estimar distância
    for group in grouped_boxes:
        # Calcular a caixa delimitadora média do grupo
        x1_avg = np.mean([box[0] for box in group])
        y1_avg = np.mean([box[1] for box in group])
        x2_avg = np.mean([box[2] for box in group])
        y2_avg = np.mean([box[3] for box in group])
        cls_avg = group[0][5]  # Assumir que todas as caixas do grupo são da mesma classe

        label = f'{model.names[int(cls_avg)]}'

        # Calcular altura média do objeto em pixels
        object_height_in_pixels = y2_avg - y1_avg
        
        # Usar a altura conhecida da classe para estimar a distância
        known_height = KNOWN_HEIGHTS[int(cls_avg)]
        if object_height_in_pixels > 0:
            distance = (known_height * FOCAL_LENGTH) / object_height_in_pixels
            distance_label = f'Distance: {distance:.2f}m'
            
            # If distance is less than or equal to 10 meters, show a warning sign
            if distance <= 10.0:
                color = (0, 0, 255)  # Red color in BGR
                show_warning(frame)
            else:
                color = (255, 0, 0)  # Blue color in BGR

            # Desenhar a caixa delimitadora média
            cv2.rectangle(frame, (int(x1_avg), int(y1_avg)), (int(x2_avg), int(y2_avg)), color, 1)
            cv2.putText(frame, label, (int(x1_avg), int(y1_avg) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 1) 
            cv2.putText(frame, distance_label, (int(x1_avg), int(y2_avg) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 1)

    # Mostrar o quadro processado
    cv2.imshow('Detecção de Animais', frame)

    # Adicionar o quadro processado ao vídeo de saída
    out.write(frame)

    # Verificar se a tecla 'q' foi pressionada para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
cap.release()
out.release()
cv2.destroyAllWindows()
