from ultralytics import YOLO
import cv2
import tkinter as tk
from threading import Thread
import time

# 모델 로드
model1 = YOLO("@@@")

# 평균 얼굴 너비 (cm)
KNOWN_WIDTH = 15.0
FOCAL_LENGTH = 600  # 보정된 초점 거리
DISTANCE_THRESHOLD = 30  # 경고 거리 기준 (cm)

popup_shown = False  # 팝업 상태 추적용

def calculate_distance(focal_length, real_width, width_in_frame):
    return (real_width * focal_length) / width_in_frame

def show_popup():
    global popup_shown
    if popup_shown:
        return  # 이미 표시 중이면 무시

    popup_shown = True
    root = tk.Tk()
    root.title("경고")
    root.geometry("300x100+600+300")
    label = tk.Label(root, text="화면과 너무 가까이 있습니다!", font=("Arial", 12), fg="red")
    label.pack(expand=True)

    # 1초 후 팝업 자동 닫기
    def close():
        time.sleep(5)
        root.destroy()
        global popup_shown
        popup_shown = False

    Thread(target=close).start()
    root.mainloop()

# 웹캠 열기
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model1(frame)

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width = x2 - x1
                distance_cm = calculate_distance(FOCAL_LENGTH, KNOWN_WIDTH, width)

                # 경고 거리 이하면 팝업 실행 (스레드로 실행)
                if distance_cm < DISTANCE_THRESHOLD and not popup_shown:
                    Thread(target=show_popup).start()

                # 화면에 표시
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{distance_cm:.2f} cm", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Face Distance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
