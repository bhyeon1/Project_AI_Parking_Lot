import cv2
import time
import numpy as np
from ultralytics import YOLO
from gpiozero import DigitalOutputDevice, DigitalInputDevice
import warnings
from PIL import Image, ImageFont, ImageDraw
from collections import deque
from enum import Enum

# ── YOLO 설정 ──
MODEL_PATH       = 'best_find_empty_place.pt'
CAM_INDEX        = 0

# ── GPIO 핀 설정 ──
NO_EMPTY_SIGNAL_PIN = 14
FIND_SIGNAL_PIN     = 15

# ── 디스플레이 설정 ──
DISABLE_GUI = False
FONT_PATH   = "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"

# ── 상태 정의 ──
class State(Enum):
    IDLE = 0
    SENSE_WAIT = 1
    GUIDE_SLOT = 2
    GUIDE_FULL = 3
    GUIDE_DONE = 4

state = State.IDLE
state_start_time = time.time()
last_target_message = "입차 대기중..."
font_loaded = False
target_history = deque(maxlen=10)
guide_threshold = 6

# ── 경고 억제 ──
warnings.filterwarnings("ignore", category=UserWarning)

# ── GPIO 초기화 ──
no_empty_signal = DigitalOutputDevice(NO_EMPTY_SIGNAL_PIN, active_high=True, initial_value=False)
find_signal     = DigitalInputDevice(FIND_SIGNAL_PIN, pull_up=False)

# ── YOLO 모델 로드 ──
model = YOLO(MODEL_PATH, verbose=False)

# ── 슬롯 매핑 함수 ──
def init_slot_mapping(img):
    res = model(img, conf=0.25, iou=0.45, imgsz=640, verbose=False)[0]
    raw = []

    if not hasattr(res, 'boxes') or not res.boxes:
        print("❗ 초기 프레임에서 슬롯을 찾을 수 없습니다.")
        return {}

    for b in res.boxes:
        cls = int(b.cls[0])
        name = res.names[cls]
        if name not in ('space-empty', 'space-occupied'):
            continue
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        cx, cy = ((x1 + x2) / 2, (y1 + y2) / 2)
        raw.append({'box': (x1, y1, x2, y2), 'center': (cx, cy)})

    raw.sort(key=lambda x: x['center'][1])
    top_row, bottom_row = raw[:17], raw[17:]
    top_row.sort(key=lambda x: x['center'][0])
    bottom_row.sort(key=lambda x: x['center'][0])

    slots = {}
    for i, slot in enumerate(top_row):
        slots[f"A{i+1}"] = slot
    for i, slot in enumerate(bottom_row):
        slots[f"B{i+1}"] = slot

    return slots

# ── 웹캠 시작 ──
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다.")

time.sleep(1.0)
ret, init_frame = cap.read()
static_slots = init_slot_mapping(init_frame)
print("슬롯 매핑 완료:", list(static_slots.keys()))

# ── 폰트 불러오기 ──
try:
    font = ImageFont.truetype(FONT_PATH, 28)
    font_loaded = True
except:
    font = ImageFont.load_default()
    print("❗ 한글 폰트를 불러올 수 없습니다.")

prev_no_empty_value = no_empty_signal.value

# ── 메인 FSM 루프 ──
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임 읽기 실패")
            break

        now = time.time()

        # ── YOLO로 빈자리 확인 ──
        empties = []
        results = model(frame, conf=0.25, iou=0.45, imgsz=640)[0]
        if hasattr(results, 'boxes') and results.boxes:
            for b in results.boxes:
                cls_id = int(b.cls[0])
                name = results.names[cls_id]
                if name != 'space-empty':
                    continue
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                cx, cy = ((x1 + x2) / 2, (y1 + y2) / 2)
                best_id, best_d2 = None, float('inf')
                for sid, slot in static_slots.items():
                    dx, dy = cx - slot['center'][0], cy - slot['center'][1]
                    d2 = dx*dx + dy*dy
                    if d2 < best_d2:
                        best_d2, best_id = d2, sid
                if best_id:
                    empties.append(best_id)

        # ── 빈자리 유무에 따라 GPIO 출력 (1 = 만차) ──
        should_be_on = not bool(empties)
        if should_be_on != prev_no_empty_value:
            no_empty_signal.value = should_be_on
            prev_no_empty_value = should_be_on

        # ── FSM 상태 전이 ──
        if state == State.IDLE:
            if find_signal.is_active:
                state = State.SENSE_WAIT
                state_start_time = now
                print("차량 감지 시작 → SENSE_WAIT 전환")

        elif state == State.SENSE_WAIT:
            if not find_signal.is_active:
                state = State.IDLE
                print("감지 해제 → IDLE 복귀")
            elif (now - state_start_time) >= 2.0:
                if empties:
                    state = State.GUIDE_SLOT
                else:
                    state = State.GUIDE_FULL
                state_start_time = now

        elif state == State.GUIDE_SLOT:
            def slot_priority(sid):
                row = 0 if sid.startswith('A') else 1
                col = int(sid[1:])
                return (col, row)
            target = min(empties, key=slot_priority)
            target_history.append(target)
            if target_history.count(target) >= guide_threshold:
                last_target_message = f"안내 → {target} 로 이동하세요."
                state = State.GUIDE_DONE
                state_start_time = now
                print("✅ 안내 시작:", last_target_message)

        elif state == State.GUIDE_FULL:
            last_target_message = "빈 자리가 없습니다!"
            state = State.GUIDE_DONE
            state_start_time = now
            print("⚠ 만차 안내")

        elif state == State.GUIDE_DONE:
            if (now - state_start_time) > 5.0:
                last_target_message = "입차 대기중..."
                target_history.clear()
                state = State.IDLE
                print("메시지 종료 → IDLE 복귀")

        # ── 메시지 결정 ──
        message = last_target_message

        # ── 메시지 렌더링 및 디스플레이 ──
        if not DISABLE_GUI:
            msg_img = np.zeros((100, 400, 3), dtype=np.uint8)
            msg_pil = Image.fromarray(msg_img)
            draw = ImageDraw.Draw(msg_pil)
            draw.text((10, 30), message, font=font, fill=(0, 255, 0))
            msg_img = np.array(msg_pil)

            cv2.imshow("Parking Guide Message", msg_img)
            cv2.imshow("Parking Guide", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        time.sleep(0.1)

finally:
    cap.release()
    cv2.destroyAllWindows()
    no_empty_signal.off()
    print("시스템 종료: 리소스 정리 완료")

