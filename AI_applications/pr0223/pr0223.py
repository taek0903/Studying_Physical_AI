import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
from ultralytics import YOLO

print(f'PyTorch: {torch.__version__}')
print(f'Device: {"cuda" if torch.cuda.is_available() else "cpu"}')

model = YOLO('yolov8n.pt')

print(f'Number of model parameters: {sum(p.numel() for p in model.model.parameters())}')

class_names = model.names
print(f'총 클래스 수 {len(class_names)}')
print(f'일부 클래스 수 {list(class_names.values())[:10]}')

# 테스트 이미지 준비
def download_image(url):
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return img
    
    except Exception as e:
        print(f'Error downloading image: {e}')
        return None
    
test_images = {
    'street': 'https://ultralytics.com/images/bus.jpg',
    'people': 'https://ultralytics.com/images/zidane.jpg',
    'animals': 'https://images.unsplash.com/photo-1583337130417-3346a1be7dee?w=640'   
}

images = {}
for name, url in test_images.items():
    img = download_image(url)
    if img is not None:
        images[name] = img
        print(f'    {name}: {img.size}')

# 샘플 이미지 시각화
if images:
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    if len(images) == 1:
        axes = [axes]

    for ax, (name, img) in zip(axes, images.items()):
        ax.imshow(img)
        ax.set_title(f'{name.capitalize()} Image', fontsize=12, fontweight='bold')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

print(images.keys())

img_name = list(images.keys())[0]
test_img = images[img_name]
print(test_img)

# yolo v8 추론
results = model(test_img)

# 결과 추출
result = results[0]     # 첫번째 결과
boxes = result.boxes

print(result.boxes)     # 검출된 객체수

# 검출 결과 상세 정보
if len(boxes) > 0:
    print('검출상세정보')
    for i, box in enumerate(boxes):
        # 박스 정보
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().item()
        cls = int(box.cls[0].cpu().item())
        class_name = class_names[cls]

        print(f' {i+1}. {class_name}: {conf:3f}'
              f' [{x1:.3f}, {y1:3f}, {x2:3f}, {y2:3f}]')
        
'''
YOLO v8 결과 해석하는 법

형식,명칭,설명
xywh,Center Coordinates,"상자의 **중심점(x, y)**과 너비(w), 높이(h) (픽셀 단위)"
- xywh: tensor([[413.9370, 494.0588, 782.1312, 525.5631]
xywhn,Normalized Center,위 데이터를 이미지의 전체 크기(1.0) 대비 비율로 정규화한 값
- xywhn: tensor([[0.5110, 0.4575, 0.9656, 0.4866],
xyxy,Corner Coordinates,"상자의 **좌측 상단(x1, y1)**과 우측 하단(x2, y2) (픽셀 단위)"
- xyxy: tensor([[2.2871e+01, 2.3128e+02, 8.0500e+02, 7.5684e+02],
xyxyn,Normalized Corner,위 데이터를 이미지 전체 크기 대비 비율로 정규화한 값
- xyxyn: tensor([[2.8236e-02, 2.1415e-01, 9.9383e-01, 7.0078e-01],
'''

from cv2.gapi import BGR2RGB
# 결과 시각화(기본)

result_img = result.plot()  # BGR

# RGB로 변환
plt.figure(figsize=(12,8))
plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title(f'YOLOv8 Detection Result ({img_name})',
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# confidence threshold 조정
thresholds = [0.25, 0.5, 0.75]
fig, axes = plt.subplots(1,3, figsize=(18,6))

for ax, conf_thresh in zip(axes, thresholds):
    # threshold 적용하며 추론
    results_thresh = model(test_img, conf=conf_thresh)
    result_thresh = results_thresh[0]

    # 시각화
    img_plot = result_thresh.plot()
    ax.imshow(cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB))
    ax.set_title(f'Confidence >= {conf_thresh}\n'
                 f'({len(result_thresh.boxes)} object)',
                 fontsize=14, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.show()

# 여러 이미지 일괄 검출

fig, axes = plt.subplots(len(images), 2, figsize=(14, 6*len(images)))

if len(images) == 1:
    axes = axes.reshape(1,-1)

for idx, (name, img) in enumerate(images.items()):
    # 원본 이미지
    axes[idx, 0].imshow(img)
    axes[idx, 0].set_title(f'{name.capitalize()} - Original Image',
                           fontsize=12, fontweight='bold')
    axes[idx, 0].axis('off')

    # 검출 결과
    results = model(img, conf=0.5)
    result_img = results[0].plot()
    axes[idx, 1].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    axes[idx, 1].set_title(f'{name.capitalize()} - Detection Result({len(result[0].boxes)} objects)',
                           fontsize=12, fontweight='bold')
    axes[idx, 1].axis('off')

plt.tight_layout()
plt.show()

# 클래스 별 통계 분석

# 모든 이미지에서 검출 수행
all_detections = {}

for name, img in images.items():
    results = model(img, conf=0.5)
    boxes = results[0].boxes

    detections = []
    for box in boxes:
        cls = int(box.cls[0].cpu().item())
        conf = box.conf[0].cpu().item()
        detections.append({
            'class': class_names[cls],
            'confidence' : conf
        })

    all_detections[name] = detections

# 클래스별 카운트
from collections import Counter

for name, detections in all_detections.items():
    print(f'\n{name.upper()} 이미지:')
    class_counts = Counter([det['class'] for det in detections])
    for cls, count in class_counts.most_common():
        avg_conf = np.mean([det['confidence'] for det in detections if det['class'] == cls])
        print(f' - {cls}: {count}개 (평균 confidence: {avg_conf:.3f})')

# 커스텀 시각화
def draw_custom_boxes(image, results, conf_threshold=0.5):
    img_np = np.array(image).copy()
    boxes = results[0].boxes

    # 색상 팔레트 (클래스별)
    np.random.seed(42)
    colors = {cls: tuple(np.random.randint(0, 255, 3).tolist())
              for cls in range(len(class_names))}

    for box in boxes:
        conf = box.conf[0].cpu().item()
        if conf < conf_threshold:
            continue

        # 박스 정보
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cls = int(box.cls[0].cpu().item())
        class_name = class_names[cls]
        color = colors[cls]

        # 박스 그리기
        cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 3)

        # 레이블 배경
        label = f'{class_name} {conf:.2f}'
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img_np, (x1, y1 - text_h - 10),
                     (x1 + text_w, y1), color, -1)

        # 레이블 텍스트
        cv2.putText(img_np, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return img_np

# 커스텀 시각화 적용
img_name = list(images.keys())[0]
test_img = images[img_name]
results = model(test_img)

custom_img = draw_custom_boxes(test_img, results, conf_threshold=0.5)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].imshow(test_img)
axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(custom_img)
axes[1].set_title('Custom Visualization', fontsize=14, fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.show()

# 배치 추론(여러 이미지 동시 처리)
import time

# 이미지 리스트 준비
img_list = list(images.values())

# 단일 추론
start_time = time.time()
for img in img_list:
    _ = model(img, verbose=False)
single_time = time.time() - start_time

# 배치 추론
start_time = time.time()
_ = model(img_list, verbose=False)
batch_time = time.time() - start_time

print(f"단일 추론: {single_time:.3f}초 ({len(img_list)}개 이미지)")
print(f"배치 추론: {batch_time:.3f}초 ({len(img_list)}개 이미지)")
print(f"속도 향상: {single_time/batch_time:.2f}배")

import tkinter as tk
from tkinter import filedialog

# 1. Tkinter 초기화 및 숨기기 (불필요한 빈 창이 뜨는 것을 방지)
root = tk.Tk()
root.withdraw()

print("\n파일 선택 창이 열립니다...")

img_path = filedialog.askopenfilename(
    title="테스트할 이미지를 선택하세요",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
)

# 3. 파일이 정상적으로 선택되었는지 확인
if img_path:
    # 이미지를 RGB 포맷으로 불러오기
    user_img = Image.open(img_path).convert('RGB')

    print(f"이미지 업로드 완료: {img_path}")
    print(f"크기: {user_img.size}")

    # 검출 수행 (model 객체가 미리 로드되어 있어야 함)
    results = model(user_img, conf=0.5)
    result_img = results[0].plot()

    # 결과 시각화
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(user_img)
    axes[0].set_title('Uploaded Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'The result: ({len(results[0].boxes)} object(s))', 
                      fontsize=14, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show() # VS Code에서는 이 코드가 실행되면 새로운 이미지 창이 팝업으로 뜹니다.

    # 검출 상세 정보 출력
    print("\n검출된 객체:")
    for i, box in enumerate(results[0].boxes):
        cls = int(box.cls[0].cpu().item())
        conf = box.conf[0].cpu().item()
        print(f"  {i+1}. {class_names[cls]}: {conf:.3f}")
else:
    print("파일 선택이 취소되었거나 이미지가 없습니다.")