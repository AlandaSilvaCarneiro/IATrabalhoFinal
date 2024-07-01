import cv2
import torch

def load_yolo_model():
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model
    except Exception as e:
        print(f"Erro ao carregar o modelo YOLO: {e}")
        return None

def detect_objects(model, frame):
    try:
        results = model(frame)
        return results
    except Exception as e:
        print(f"Erro ao detectar objetos: {e}")
        return None

def draw_boxes(results, frame):
    if results is not None:
        for *box, conf, cls in results.xyxy[0].tolist():
            label = f'{results.names[int(cls)]} {conf:.2f}'
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

def main():
    model = load_yolo_model()
    if model is None:
        print("Falha ao carregar o modelo. Verifique os erros acima.")
        return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Erro ao acessar a câmera.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar frame da câmera.")
            break

        results = detect_objects(model, frame)
        frame = draw_boxes(results, frame)

        cv2.imshow('YOLO Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
