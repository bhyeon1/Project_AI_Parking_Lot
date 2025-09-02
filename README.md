# 스마트 AI 기반 주차장 시스템

AI 모델(YOLOv5 + OCR)을 활용해 차량 번호판을 인식하고, 등록 여부 확인 및 주차 공간 상태를 실시간으로 판단하여 사용자에게 **주차 안내** 및 **경고 메시지**를 제공하는 지능형 스마트 주차장 시스템입니다. 

본 프로젝트는 **Google Colab**, **Ubuntu 기반 Raspberry Pi**, **카메라**, **YOLOv5**, **OCR (CRNN+CTC)** 기반으로 설계되었으며, 실제 차량 입고 및 주차공간 상태를 반영하여 동작합니다.

---
## 고지
- 해당 프로젝트는 기능마다 branch에 분리되어 수록되어있습니다. 추후 main에 merge 될 수 있습니다. 


## 시스템 구성도

- **카메라 모듈**  
  - 차량 진입 시 번호판 촬영
  - 주차장 공간 모니터링

- **YOLOv5 모델 (license plate detection)**  
  - 번호판 위치 탐지
  - 주차된 차량/빈 자리 탐지 (추후 확장)

- **OCR 모델 (CRNN + CTC)**  
  - 번호판 문자 인식
  - 등록 차량 여부 확인

- **Raspberry Pi + Ubuntu**  
  - AI inference (YOLO + OCR)
  - 주차장 UI 안내 및 음성 경고 (추후 확장 가능)

---

## 프로젝트 구조

```text
.
├── yolov5/                     # YOLOv5 설치 및 학습 코드
│   ├── runs/train/             # 학습 결과 저장 (best.pt)
│   └── runs/detect/            # 탐지 결과 저장
├── car_yolo_yolov5/           
│   ├── images/train/           # 훈련 이미지
│   ├── labels/train/           # YOLO 라벨 (.txt)
│   └── data.yaml               # 데이터셋 구성
├── car_num_unzip/             
│   └── car_num_img/            # OCR 학습 이미지
│   └── car_num_json/           # OCR JSON 라벨
└── ocr_model.py                # CRNN + CTC OCR 코드
```

---

## 모델 1: 차량 번호판 탐지 (YOLOv5)

- **데이터셋 구성**
  - `.json` → YOLO 형식 `.txt`로 자동 변환
  - `/images/train/`, `/labels/train/` 생성됨

- **학습 명령어**
```bash
python train.py \
  --img 640 \
  --batch 16 \
  --epochs 50 \
  --data car_yolo_yolov5/data.yaml \
  --weights yolov5s.pt \
  --name plate_yolo_train
```

- **탐지 명령어**
```bash
python detect.py \
  --weights runs/train/plate_yolo_train/weights/best.pt \
  --img 640 \
  --source /content/test_imgs \
  --save-txt \
  --save-crop
```

---

## 모델 2: 차량 번호 OCR (CRNN + CTC)

- **구성**
  - CRNN (CNN + BiLSTM + FC)
  - CTC Loss
  - 데이터 전처리: `Resize → Normalize → Augment`

- **학습 성능**
  - Accuracy: 85% 이상 (Val)
  - Precision/Recall 지속 증가
  - 조기 종료(EarlyStopping)로 안정적 학습

- **결과 예시**
```text
[GT] 01가2178 → [Pred] 01가2178
[GT] 01가2525 → [Pred] 01가2525
[GT] 01가3042 → [Pred] 01가3042
```

---

## 버전 호환성 (Colab 환경 기준)

<img src="https://user-images.githubusercontent.com/your-id/680e1e45-2bcd-4e35-a73a-7be38e503987.png" width="600">

- YOLO 모델은 `torch==2.1.0 + cu118`, `numpy==1.24.3` 환경에서 최적
- OCR 모델은 `torch==2.6.0 + cu124`, `numpy==2.0.2` 기반
- **서로 다른 CUDA 버전 사용으로 GPU 동시 운용 시 충돌 주의**

---

## 실행 순서 요약

1. `YOLOv5` 학습 및 best.pt 추출
2. 번호판 영역만 crop 후 OCR 데이터로 사용
3. `OCR 모델` 학습 및 저장
4. 실시간 영상에서 `YOLO → OCR` 추론 → 번호 비교
5. **등록 차량 여부 판별 후 입장 허용**
6. **주차 공간 상태를 YOLO로 판단 → 안내 or 경고**

---

## 향후 발전 방향

- [ ] YOLOv5를 활용한 주차선, 주차공간 탐지 기능 추가
- [ ] 삐딱한 주차 판단을 위한 시멘틱 세그멘테이션 (SegFormer 등) / RBox 사용으로 객체의 방향과 맞는 box 구현 but 데이터 셋 변환 필요
- [ ] Web UI 및 라즈베리파이 음성 안내 연동
- [ ] 실시간 입출차 데이터 기록 + DB 연동

---

## 라이선스

본 프로젝트는 연구 및 학습 목적이며, 상업적 이용은 별도 문의 바랍니다.

---

## 기여자

- 김을중
- 김한벗
- 김병현
- 최규리

---
