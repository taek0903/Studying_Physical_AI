import torch
from torch import tensor
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torchviz import make_dot
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image, ImageDraw, ImageFont
import urllib.request
import requests
import cv2
import torchvision
from io import BytesIO

plt.rcParams["font.family"] = "Malgun Gothic"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'PyTorch: {torch.__version__}')
print(f'TorchVision: {torchvision.__version__}')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# COCO 클래스 이름 정의 (80개)
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# IOU
def compute_iou(box1, box2):
    # box format [x1, y1, x2, y2]

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 교집합 영역
    intersection = max(0, x2-x1) * max(0, y2-y1)

    # 각 박스 면적
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])

    # 합집합 영역
    union = area1 + area2 - intersection

    # IOU
    iou = intersection / union if union > 0 else 0

    return iou

# IOU 테스트
box_gt = [100 ,100, 200, 200]
box_pred1 = [120, 120, 220, 220]
box_pred2 = [150, 150, 250, 250]
box_pred3 = [250, 250, 350, 350]

print(f'ground_truth: {box_gt}')
print(f'prediction: {box_pred1}, IOU: {compute_iou(box_gt, box_pred1):.3f}')
print(f'prediction: {box_pred2}, IOU: {compute_iou(box_gt, box_pred2):.3f}')
print(f'prediction: {box_pred3}, IOU: {compute_iou(box_gt, box_pred3):.3f}')

# NMS 함수 구현
'''
NMS : 중복되는 bbox(bounding box) 제거 (겹치는 박스 중 가장 신뢰도 높은 score 높은 박스만 남기겠다.)
boxes = (N,4) [x1, y1, x2, y2]
scores = [N, ] - confidence scores
'''

def simple_nms(boxes, scores, threshold=0.5):
    # scroe(신뢰도) 내림차순 정렬
    indices = np.argsort(scores)[::-1]  # 역순
    keep = []
    # 가장 높은 신뢰도 점수를 보이는 최종 선택된 인덱스 저장
    while len(indices) > 0:
        current = indices[0]
        # indices[0] : 내림차순 정렬되어 있으니깐 가장 높은 신뢰도(score) 가진 박스
        keep.append(current)

        if len(indices) == 1 :
            break
        '''
        indices의 개수가 0보다 크면 반복문 작동
        keep 리스트 에 inices[0]의 값을 추가
        indicecs의 개수가 1일 경우 함수를 탈출 (이터레이터를 호출로 인해 하나씩 줄어듦)
        '''
        
        # 나머지 box들과 IOU 계산
        current_box = boxes[current]
        remaining_boxes = boxes[indices[1:]]

        # IOU가 threshold 이하인 box만 유지
        ious = np.array([compute_iou(current_box, box) for box in remaining_boxes])
        indices = indices[1:][ious <= threshold]
        '''
        currnet_box 변수에 1등 박스 정보를 저장한다.
        remaining_boxes 변수에 역순으로 정렬된 스코어 인덱스를 통해 
        1등을 제외한 나머지 박스들을 저장(N-1개)
        ious 를 통해 나머지 박스들의 iou값을 계산
        1등을 제외하고 ious의 값이 threshold(임계값) 이하인 것만 남김 
        '''
    
    return keep

test_boxes = np.array([
    [100,100,200,200],
    [110,110,210,210],
    [105,105,205,205],
    [300,300,400,400]
])

test_scores = np.array([0.9,0.8,0.85,0.95])

print(f'input_boxes: {len(test_boxes)}개')
keep_indices = simple_nms(test_boxes, test_scores, threshold=0.5)
print(f'nms 후 {len(keep_indices)}개 유지')
print(f'유지된 인덱스 {keep_indices}')
print(f'유지된 scores {test_scores[keep_indices]}')

# Faster R-CNN 모델
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights).to(device)

model.eval()
preprocess = weights.transforms()

# 테스트 이미지 준비
def create_sample_image():
    '''샘플 이미지 생성 (사람, 의자, 책 그리기)'''
    img = Image('RGB', (800, 600), color=(240,240,240))
    draw = ImageDraw.Draw(img)

    # 사람 그리기 (간단한 스틱맨)
    # 머리
    draw.ellipse([150, 100, 250, 200], fill=(255, 220, 180), outline=(0, 0, 0), width=3)
    # ellipse:  원형 outline=(0, 0, 0) 윤곽선 검정색
    # 몸통
    draw.rectangle([180, 200, 220, 400], fill=(0, 100, 200), outline=(0, 0, 0), width=3)
    # 팔
    draw.line([180, 250, 120, 300], fill=(0, 100, 200), width=15)
    draw.line([220, 250, 280, 300], fill=(0, 100, 200), width=15)
    # 다리
    draw.line([180, 400, 140, 550], fill=(50, 50, 50), width=15)
    draw.line([220, 400, 260, 550], fill=(50, 50, 50), width=15)

    # 의자 그리기
    draw.rectangle([500, 300, 650, 350], fill=(139, 69, 19), outline=(0, 0, 0), width=3)
    draw.rectangle([520, 350, 540, 500], fill=(139, 69, 19), outline=(0, 0, 0), width=3)
    draw.rectangle([610, 350, 630, 500], fill=(139, 69, 19), outline=(0, 0, 0), width=3)
    draw.rectangle([510, 150, 640, 300], fill=(160, 82, 45), outline=(0, 0, 0), width=3)

    # 책 그리기
    draw.rectangle([350, 450, 450, 550], fill=(200, 50, 50), outline=(0, 0, 0), width=3)
    draw.line([400, 450, 400, 550], fill=(0, 0, 0), width=2)

    # 텍스트 추가
    draw.text((300, 30), "Sample Detection Image", fill=(0, 0, 0))

    return img

def download_image_safe():

    urls = [
        "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg",
        "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg",
        "https://ultralytics.com/images/zidane.jpg"
    ]

    for url in urls:
        try:
            print(f"다운로드 시도: {url[:50]}...")
            response = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content)).convert("RGB")
                print(f"이미지 다운로드 성공!")
                return img
        except Exception as e:
            print(f"  실패: {str(e)[:50]}")
            continue

    return None

img = download_image_safe()

if img is None:
    # 온라인 다운로드 실패 -> 샘플 이미지 생성
    img = create_sample_image()

else:
    if max(img.size) > 1000:
        img.thumbnail((1000,1000))
        print(f'이미지 리사이즈: {img.size}')

plt.figure(figsize=(10, 6))
plt.imshow(img)
plt.axis('off')
plt.title('테스트 이미지')
plt.tight_layout()
plt.show()

print(f"이미지 크기: {img.size}")

img_tensor = preprocess(img).unsqueeze(0).to(device)

print(img_tensor.shape)

with torch.no_grad():
    predictions = model(img_tensor)[0]

print(predictions.keys())

boxes = predictions['boxes'].cpu().numpy()
labels = predictions['labels'].cpu().numpy()
scores = predictions['scores'].cpu().numpy()

print(f'검출된 객체수(전체): {len(boxes)}')

# 검출 결과가 있을 때만 통계 출력
if len(scores) > 0:
    print(f'confidence 범위 : {scores.min():.2f} ~ {scores.max():.2f}')
else:
    print('검출된 객체가 없어요.')

# 결과 필터링 및 시각화
# confidence threshold로 필터링

conf_threshold = 0.5

if len(boxes) > 0:
    mask = scores >= conf_threshold
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    filtered_labels = labels[mask]
    '''
    넘파이 불리언 마스크를 사용하여 필터링
    True 값만 남기고 False 값은 걸러내라
    따라서 scores의 점수가 coonf_threshold 보다 작으면 걸러내라 
    '''
    print(f'필터링 한 후 객체 수: {len(filtered_boxes)}')

    if len(filtered_boxes) > 0:
        # 시각화
        img_np = np.array(img)
        img_draw = img_np.copy()

        # 색상 정의
        colors = plt.cm.tab20(np.linspace(0, 1, 20))

        for i, (box, score, label) in enumerate(zip(filtered_boxes, filtered_scores, filtered_labels)):
            x1, y1, x2, y2 = box.astype(int)
            class_name = COCO_CLASSES[label]

            # 박스 그리기
            color = tuple((np.array(colors[label % 20][:3]) * 255).astype(int).tolist())
            cv2.rectangle(img_draw, (x1,y1), (x2,y2), color, 3)

            # 레이블 그리기
            text =f'{class_name}: {score:.2f}'
            (text_w, text_h), _ =cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_draw, (x1, y1 - text_h -10), (x1 + text_w, y1), color, -1)
            # -1: 내부를 FILL 더 채워주세요

            cv2.putText(img_draw, text, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            # (255,255,255) 흰색
            print(f'{i+1}, {class_name} (confidence: {score:.2f}), Box: [{x1}, {y1}, {x2}, {y2}]')

        plt.figure(figsize=(12, 8))
        plt.imshow(img_draw)
        plt.axis('off')
        plt.title(f'Fast RCNN 검출결과(Threshold: {conf_threshold})')
        plt.tight_layout()
        plt.show()
    else:
        print(f'Threshold {conf_threshold} 이상인 객체가 없습니다.')
else:
    print('검출할 객체가 없습니다.')

# 다양한 Threshold 비교
if len(boxes) > 0:
    thresholds = [0.3, 0.5, 0.7, 0.9]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()

    img_np = np.array(img)
    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    for idx, thresh in enumerate(thresholds):
        mask = (scores >= thresh)
        img_temp = img_np.copy()

        temp_boxes = boxes[mask]
        temp_scores = scores[mask]
        temp_labels = labels[mask]

        for box, score, label in zip(temp_boxes, temp_scores, temp_labels):
            x1, y1, x2, y2 = box.astype(int)
            class_name = COCO_CLASSES[label]
            color = tuple((np.array(colors[label % 20][:3]) * 255).astype(int).tolist())

            cv2.rectangle(img_temp, (x1, y1), (x2, y2), color, 2)
            text = f'{class_name}: {score:.2f}'
            cv2.putText(img_temp, text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        axes[idx].imshow(img_temp)
        axes[idx].axis('off')
        axes[idx].set_title(f'Threshold: {thresh} ({len(temp_boxes)} 객체)', fontsize=12)
        print(f"Threshold {thresh}: {len(temp_boxes)}개 검출")

    plt.tight_layout()
    plt.show()

# 검출 통계 분석
if len(boxes) > 0 and len(boxes[scores >= conf_threshold]) > 0:

    # 코드 작성
    # 기준점(임계치) threshold 이상을 충족(True)하는 박스만 선택
    # >> class label과 score 만 선택해줘
    mask = (scores >= conf_threshold) # T/F 출력
    filtered_labels = labels[mask]
    filtered_scores = scores[mask]

    # 클래스 별 통계
    from collections import Counter
    class_counts = Counter(filtered_labels)

    print(f"\n클래스별 검출 수 (Threshold >={conf_threshold}):")

    for label_id, count in class_counts.most_common():
        class_name = COCO_CLASSES[label_id]
        avg_conf = filtered_scores[filtered_labels == label_id].mean()
        print(f"{class_name}: {count}개 (평균 confidence: {avg_conf:.3f})")

    # Confidence 분포 시각화
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.hist(scores, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(x=conf_threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold: {conf_threshold}')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.title('전체 검출 Confidence 분포')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    class_names = [COCO_CLASSES[l] for l in filtered_labels]
    unique_classes = list(set(class_names))
    class_scores = [filtered_scores[np.array(class_names) == c].mean()
                    for c in unique_classes]
    plt.barh(unique_classes, class_scores, color='skyblue', edgecolor='black')
    plt.xlabel('Average Confidence')
    plt.title('클래스별 평균 Confidence')
    plt.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.show()
'''
threshold 0.5를 기준으로 왼쪽으로 갈수록(confidence 크기 작아짐)
검출하는 객체의 수가 많다.
threshold 0.5를 기준으로 오른쪽으로 갈수록(confidence 크기 커짐)
검출하는 객체의 수가 적다.

그리고 그래프를 보면 confidence크기가 0.1~0.2 사이에서 
검출된 객체의 수가 가장 많다는 것을 확인할 수 있다.

threshold가 0.5보다 작아지면 confidence가 낮은 객체들도 판별하기 때문에
판별된 객체의 수가 많아 질 것이다.
이 말은 즉 기준인 threshold 보다 오른 쪽에 있는 객체들이 사진에 표시가 될것이다.

오른쪽 그래프는 넥타이와 사람의 평균 신뢰도가 80%를 넘는 것을 확인할 수 있고
0.5 이하의 객체들을 걸러내고 살아남은 박스들의 평균 점수가 80~90% 이상의 아주 강한
확신을 가지고 검출되었다는 결과로 볼 수 있다.
'''