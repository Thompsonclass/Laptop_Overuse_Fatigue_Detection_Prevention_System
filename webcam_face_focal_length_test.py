from ultralytics import YOLO
import cv2

# 모델 로드
model = YOLO("@@@")

# 실제 얼굴 너비(cm)와 측정용 거리(cm)
REAL_FACE_WIDTH = 15.0
KNOWN_DISTANCE = 60.0  # 줄 자로 고정

# 초점 거리 저장용 변수
focal_length_computed = None

# 얼굴 너비 추출 함수
def get_face_width_px(results):
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                return x1, y1, x2, y2, x2 - x1
    return None

# 거리 계산
def calculate_distance(focal_length, real_width, width_px):
    return (real_width * focal_length) / width_px

# 웹캠 시작
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    face_info = get_face_width_px(results)

    if face_info:
        x1, y1, x2, y2, width_px = face_info

        # 초점 거리 계산
        if not focal_length_computed:
            focal_length_computed = (width_px * KNOWN_DISTANCE) / REAL_FACE_WIDTH

        # 예측 거리 계산
        predicted_distance = calculate_distance(focal_length_computed, REAL_FACE_WIDTH, width_px)

        # 시각화
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # yolo가 계산한 얼굴 너비비
        cv2.putText(frame, f"Width(px): {width_px}", (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        # 초점 거리px(mm이 기본단위이긴한데 계산용을 위해서 임의로 px로 바꿈꿈)
        cv2.putText(frame, f"FocalLength: {focal_length_computed:.2f}px", (x1, y2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)
        # 예측 거리
        cv2.putText(frame, f"Predicted Distance: {predicted_distance:.2f} cm", (x1, y2 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Distance Estimation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
