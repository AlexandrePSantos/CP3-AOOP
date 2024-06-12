from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
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

# Carregar o modelo treinado
model = YOLO('yolov9c.pt')

# Carregar uma imagem
img_path = './cows2.jpg'
img = cv2.imread(img_path)

# Verificar se a imagem foi carregada corretamente
if img is None:
    print("Erro ao carregar a imagem.")
    exit()

# Realizar a detecção
results = model(img)

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

    # Desenhar a caixa delimitadora média
    cv2.rectangle(img, (int(x1_avg), int(y1_avg)), (int(x2_avg), int(y2_avg)), (255, 0, 0), 2)
    cv2.putText(img, label, (int(x1_avg), int(y1_avg) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Calcular altura média do objeto em pixels
    object_height_in_pixels = y2_avg - y1_avg
    
    # Usar a altura conhecida da classe para estimar a distância
    known_height = KNOWN_HEIGHTS[int(cls_avg)]
    if object_height_in_pixels > 0:
        distance = (known_height * FOCAL_LENGTH) / object_height_in_pixels
        distance_label = f'Distance: {distance:.2f}m'
        cv2.putText(img, distance_label, (int(x1_avg), int(y2_avg) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Converter BGR para RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Exibir a imagem
plt.imshow(img_rgb)
plt.axis('off')
plt.show()
