import cv2
import threading
import time
import tkinter as tk
from datetime import datetime
from ultralytics import YOLO
import os

log_file = "log_test.txt"
log_interval = 5  # 5초 간격 저장
last_log_time = 0

# 로그 파일 초기화
with open(log_file, "w") as f:
    f.write("시간,깜빡임 수,하품 수,거리 경고 수,피로도 점수\n")

# 설정 
FOCAL_LENGTH = 660 # 카메라 초점 거리 (픽셀, 테스트 검증 값)
KNOWN_WIDTH = 15.0 # 평균 얼굴 너비 (cm)
DISTANCE_THRESHOLD = 30 # 경고 기준 거리 (cm)

#  상태 변수 
frame_lock = threading.Lock()
latest_frame = None
running = True
popup_active = False

blink_count = 0
yawn_count = 0
warning_count = 0
start_time = time.time()
MAX_DURATION = 3 * 60 * 60 #3시간
# debounce
last_blink_time = 0
last_yawn_time = 0
cooldown = 1.0  # 1초 간격

# 거리 계산 
def calculate_distance(focal_length, real_width, width_in_frame):
    return (real_width * focal_length) / width_in_frame

# 팝업 함수
def show_warning_popup():
    global popup_active
    popup_active = True
    root = tk.Tk()
    root.title("경고")
    root.geometry("300x150+600+300")
    label = tk.Label(root, text=" 너무 가까이 있습니다! (30cm 미만)", font=("Arial", 12), fg="red")
    label.pack(pady=20)

    def close():
        root.destroy()
        global popup_active
        popup_active = False

    button = tk.Button(root, text="확인", command=close)
    button.pack(pady=10)
    root.mainloop()

#  웹캠 프레임 캡처
def capture_thread(cap):
    global latest_frame, running
    while running:
        ret, frame = cap.read()
        if not ret:
            break
        with frame_lock:
            latest_frame = frame
    cap.release()

#  메인 실행 
def main():
    global running, blink_count, yawn_count, warning_count
    global last_blink_time, last_yawn_time, last_log_time

    # 모델 로딩
    model_eye = YOLO("@@@")
    model_face = YOLO("@@@")
    model_eye.fuse()
    model_face.fuse()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    thread = threading.Thread(target=capture_thread, args=(cap,), daemon=True)
    thread.start()

    try:
        while True:
            if popup_active:
                time.sleep(0.1)
                continue
            
            if time.time() - start_time >= MAX_DURATION:
                print("3시간이 지나 자동으로 프로그램을 종료합니다.")
                break

            with frame_lock:
                if latest_frame is None:
                    continue
                frame = latest_frame.copy()

            # 얼굴 감지 및 거리 계산
            face_result = model_face.predict(source=frame, imgsz=640, verbose=False)[0]
            for box in face_result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                face_width = x2 - x1
                distance_cm = calculate_distance(FOCAL_LENGTH, KNOWN_WIDTH, face_width)

                # 거리 표시
                cv2.putText(frame, f"Distance: {distance_cm:.1f} cm", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # 경고 조건
                if distance_cm < DISTANCE_THRESHOLD and not popup_active:
                    warning_count += 1
                    threading.Thread(target=show_warning_popup, daemon=True).start()

                # 얼굴 박스
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # 눈/입 감지
            eye_result = model_eye.predict(source=frame, imgsz=640, verbose=False)[0]
            for box in eye_result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_id = int(box.cls[0].item())
                class_name = model_eye.names[class_id]

                now = time.time()
                if class_name == "Closed-Eye" and now - last_blink_time > cooldown:
                    blink_count += 1
                    last_blink_time = now
                elif class_name == "Yawn" and now - last_yawn_time > cooldown:
                    yawn_count += 1
                    last_yawn_time = now

                label = f"{class_name}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 피로도 계산
            elapsed = time.time() - start_time
            fatigue_score = (
                (blink_count * 0.7 + yawn_count * 2 + warning_count * 1.5)
                / (elapsed / 60 + 1)
            ) * 2
            fatigue_score = min(int(fatigue_score), 100)

            # 현재 시간 표시
            now_str = datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame, f"Time: {now_str}", (480, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

            # 상태 출력
            cv2.putText(frame, f"Blinks: {blink_count} | Yawns: {yawn_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.putText(frame, f"Fatigue: {fatigue_score}%", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # 결과 출력
            cv2.imshow("Fatigue Detection", frame)
            # 주기적으로 로그 저장
            if time.time() - last_log_time >= log_interval:
                last_log_time = time.time()
                with open(log_file, "a") as f:
                    f.write(f"{now_str},{blink_count},{yawn_count},{warning_count},{fatigue_score}\n")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        running = False
        thread.join(timeout=1)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
