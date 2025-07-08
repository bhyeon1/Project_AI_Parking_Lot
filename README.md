# Parking Detection & Alignment Check (YOLOv8-based)

본 프로젝트는 **YOLOv8 Object Detection**을 기반으로 다음 기능을 수행하는 Python 기반 도구입니다:

- 주차장 이미지 또는 영상에서 **주차 공간(`space-empty`)과 차량(`space-occupied`)** 감지
- 차량이 **정상적으로 주차 공간에 위치했는지 판단**
    - 기준: IoU 기반 겹침 정도 + 중심점이 주차 공간 내부에 있는지
- 결과를 이미지에 시각화하여 `GOOD`, `BAD` 레이블로 출력

---

## 프로젝트 구조

```
parking-detector/
├── best_parking_model.pt       # 학습 완료된 YOLOv8 모델
├── parking_status_checker.py   # 주차 상태 판단 코드
├── example.jpg                 # 테스트 이미지 (예시)
├── README.md
```

---

## 환경 구성

### Python 버전
- Python >= 3.8

### 필수 패키지 설치

```bash
pip install ultralytics shapely opencv-python matplotlib
```

---

## 실행 방법

### 1. 파이썬 코드로 직접 실행

```python
from parking_status_checker import analyze_parking_iou_v2

analyze_parking_iou_v2("example.jpg")
```

### 2. CLI 또는 터미널 실행 (원할 경우 argparse 버전 제공 가능)

```bash
python parking_status_checker.py --image example.jpg
```

---

## 주요 기능

- **YOLO 기반 탐지**: 공간(`space-empty`), 차량(`space-occupied`)을 정확히 감지
- **삐뚤 판별 기준**
  - 차량 박스와 주차 공간 박스의 **IoU ≥ 0.05**
  - 또는 차량의 **중심점이 주차 공간 내부**에 존재할 경우 정상(GOOD) 처리
- **시각화**: 차량 박스에 `GOOD` (초록색) 또는 `BAD` (빨간색) 텍스트와 박스 표시
- **중복 제거**: occupied된 공간은 empty로 이중 감지되지 않도록 필터링 처리 포함

---

## 학습 모델 정보

- 모델 아키텍처: YOLOv8
- 클래스:
  - `space-empty`: 비어 있는 주차 공간
  - `space-occupied`: 차량이 차지한 공간
- 학습 데이터:
  - Custom dataset (parking-lot annotated images)
  - 라벨링 형식: YOLO format (`class x_center y_center width height`)

---

## 참고 사항

- `.pt` 모델 파일과 `.py` 분석 코드는 반드시 **같은 경로**에 두어야 합니다.
- 입력 이미지는 `.jpg`, `.png` 형식을 모두 지원합니다.
- Ubuntu, Colab, Windows 모두 호환됩니다.

---

## 라이선스

본 프로젝트는 [MIT License](LICENSE) 하에 공개되어 있습니다.

---
