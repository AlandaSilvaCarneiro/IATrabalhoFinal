import cv2

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Tente com CAP_DSHOW
    if not cap.isOpened():
        print("Erro ao acessar a câmera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar frame da câmera.")
            break

        cv2.imshow('Câmera Teste', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
