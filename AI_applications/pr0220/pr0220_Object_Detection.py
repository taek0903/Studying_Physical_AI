# 환경/유틸
import torch, cv2, numpy as np, matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from PIL import Image
print('TorchVision:', torchvision.__version__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

from IPython.core import interactiveshell
'''
IoU 계산 함수 : 두 박스의 교집합/합집합 비율
IoU : 두 박스 (boxA, boxB) 얼마나 겹치는지 비율로 나타낸 값 [0,1]
0: 겹치지 않음, 1: 동일 위치
'''

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])  # 겹치는 영역 왼쪽 x좌표    
    yA = max(boxA[1], boxB[1])  # 겹치는 영역 왼쪽 y좌표    
    xB = min(boxA[2], boxB[2])  # 겹치는 영역 오른쪽 x좌표  
    yB = min(boxA[3], boxB[2])  # 겹치는 영역 오른쪽 y좌표

    inter = max(0, xB-xA) * max(0, yB-yA)
    '''
    xB-xA 겹치는 부분 W, yB-yA 겹치는 부분 H
    두 값이 모두 0보다 작으면 겹치지 않음
    '''

    # A면적
    areaA = (boxA[2]-boxB[0]) * (boxA[3]-boxB[1])
    # B면적
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

    # 합집합 영역
    union = areaA + areaB - inter

    return inter / union if union > 0 else 0    # 합집합 영역이 0이면 0을 반환(zeroDivision 방지)

# 사전 학습 Faster R-CNN 모델 로드
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT   # MS COCO 사전학습 가중치

det_model = fasterrcnn_resnet50_fpn(weights=weights).to(device)

class_names = weights.meta['categories']
print(f'총 예측 가능한 클래스 수: {len(class_names)}개')
print(class_names[:15])

det_model.eval()
preproc = weights.transforms()

img = Image.new('RGB', (640, 480), color=(200,200,200))

# matplotlib로 이미지 표현

# 사각형 그려보기
canvas = np.array(img).copy()
cv2.rectangle(canvas,
              (100,100),    # 사각형 왼쪽 위 좌표 (top-left-point)
              (300,300),    # 사각형 오른쪽 아래 좌표 (bottom-right-point)
              (255,0,0),    # BGR
              3)            # 선 두께

cv2.circle(canvas,
           (450,250),       # 원의 중심 좌표
           60,              # r(radious) 반지름
           (0,255,0),       # green
           3)

img = Image.fromarray(canvas)
cv2.imshow('test_img', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 테스트 이미지 준비 (샘플 다운로드)
# 간단한 단색/박스 테스트
# 단색 이미지는 객체가 없어서 boxes=[] => 실제 사진 사용

import urllib.request

# COCO 샘플 이미지 다운로드
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
urllib.request.urlretrieve(url, 'test_img.jpg')

# 이미지 로드

img = Image.new('RGB', (640, 480), color=(200,200,200))

canvas = np.array(img).copy()
cv2.rectangle(canvas,
              (100,100),    # 사각형 왼쪽 위 좌표 (top-left-point)
              (300,300),    # 사각형 오른쪽 아래 좌표 (bottom-right-point)
              (255,0,0),    # BGR(open cv)
              3)            # 선 두께

cv2.imshow('img', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = Image.open('test_img.jpg').convert('RGB')

canvas_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

cv2.imshow('test_img', canvas_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 전처리 => 모델 추론
# preproc = weights.transforms()

det_model.eval()

x = preproc(img).unsqueeze(0).to(device)    # 배치 차원 추가

with torch.no_grad():
    output = det_model(x)
    out = det_model(x)[0]   # 딕셔너리 형태로 출력

print(output[:10])
print(out.keys())

boxes = out['boxes'].cpu().numpy()      # box 좌표
labels = out['labels'].cpu().numpy()    # 라벨(카테고리)
scores = out['scores'].cpu().numpy()    # 신뢰도

print(boxes)
print(labels)
print(scores)

thr = 0.5   # 임계치
keep = scores >= thr
boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

# 시각화
vis = np.array(img).copy()

for (x1, y1, x2, y2), s, lb in zip(boxes, scores, labels):
    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 3)
    cv2.putText(vis, f'id{int(lb)} {s:.2f}', (int(x1), int(y1-5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,2))

vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10,8))
plt.imshow(vis_rgb)
plt.title('Faster-RCNN Prediction')
plt.axis('off')
plt.show()

print(compute_iou([100,100,300,300], [150,150,350,350]))

def iou_pytorch(boxes1, boxes2):

    # 교집합 영역 계산
    # 두 박스 왼쪽 위 (x1, y1) 중 큰 값 선택 (교차 영역에서 시작점)
    x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
    y1 = torch.max(boxes1[..., 1], boxes2[..., 1])

    '''
    boxes1[..., 1], boxes2[..., 1]
    ...: 파이썬 _(사용안함) 의미와 유사
    => 앞의 모든 차원을 그대로 두고, 마지막 차원 인덱스 0만 추출하겠다.
    '''

    # 두 박스 오른쪽 아래 (x2,y2) 중 작은 값(교차 영역에서 끝점)
    x2 = torch.min(boxes1[..., 2], boxes2[..., 2])  # w
    y2 = torch.min(boxes1[..., 3], boxes2[..., 3])  # h

    # 교차 영역의 width, height (음수가 나오면 0으로 처리 => 안겹칠 경우)
    inter_w = torch.clamp(x2 - x1, min=0)
    inter_h = torch.clamp(y2 - y1, min=0)

    # 교차 영역 면적
    inter_area = inter_w * inter_h
    
    # 각 박스 면적 계산
    area1 = (boxes1[..., 2]-boxes1[..., 0]) * (boxes1[..., 3]-boxes1[..., 1])
    area2 = (boxes2[..., 2]-boxes2[..., 0]) * (boxes2[..., 3]-boxes2[..., 1])

    # 합집합 영역
    union = area1 + area2 - inter_area

    # zero_division 방지
    eps = 1e-7

    # iou 계산
    iou = inter_area / (union * eps)
    return iou

# 단일 박스 iou
boxA = torch.tensor([10,10,60,60])
boxB = torch.tensor([30,30,80,80])

print(iou_pytorch(boxA, boxB))

print(boxA.shape)   #(4,)

# 배치(batch) iou
boxA = torch.tensor([[10,10,60,60],[50,50,100,100]])
boxB = torch.tensor([[30,30,80,80],[60,60,120,120]])

print(iou_pytorch(boxA, boxB))

print(boxA.shape)   #[2, 4]

'''
coco 데이터셋 클래스에 들어있는 객체를 찾아서 표시해주는 코드이다.
불러온 데이터는 고양이 2마리가 소파에 누워있는 사진이다.
2가지 클래스를 특정했다고 유추할 수 있는데 하나는 리모컨이고 하나는 고양이 인 것 같다.
결과를 보면 누워 있는 고양이와 리모컨을 특정햇다. 리모컨은 각 객체 하나씩 잘 특정했지만
고양이는 특정한 사각형들이 많았다. 이는 중복이 많은 거로 확인할 수 있을 것같다.
나중에 임계값을 조정하면 특정한 사각형의 개수가 줄어들 거로 보인다.
'''