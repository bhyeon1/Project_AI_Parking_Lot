import cv2
import time
from ultralytics import YOLO

# 1) 모델 로드
model = YOLO('best_find_empty_place.pt')  # space-empty / space-occupied 포함

# 2) 카메라 열기 (0: 첫 번째 연결된 캠)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다.")

# 프레임 속도 측정용
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다. 종료합니다.")
        break

    # 해상도 조절이 필요하면 여기에
    # frame = cv2.resize(frame, (1280, 720))

    # 3) 모델 추론
    results = model(frame)[0]

    empty_count = 0
    occupied_count = 0

    # 4) 박스 그리기
    for box in results.boxes:
        cls_id = int(box.cls[0])
        name = results.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        if name == 'space-empty':
            color = (0, 255, 0)   # 녹색
            empty_count += 1
        elif name == 'space-occupied':
            color = (0, 0, 255)   # 빨간색
            occupied_count += 1
        else:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 5) FPS 및 카운트 표시
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f'FPS: {fps:.1f}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, f'Empty: {empty_count}', (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f'Occupied: {occupied_count}', (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # 6) 화면 출력
    cv2.imshow('Empty Space Detection Test', frame)
    # ESC 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
