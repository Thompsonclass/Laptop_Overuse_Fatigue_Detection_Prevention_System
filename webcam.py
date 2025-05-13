import cv2
from ultralytics import YOLO

def main():
    # YOLO 모델 로드
    model_path = "@@@"
    model = YOLO(model_path)

    # 웹캠 열기
    cap = cv2.VideoCapture(0)

    # 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # YOLO 예측 수행
        results = model(frame)

        # 결과 시각화
        output_frame = results[0].plot()

        # 화면에 출력
        cv2.imshow("YOLO Webcam - Single Model", output_frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # 정리
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
