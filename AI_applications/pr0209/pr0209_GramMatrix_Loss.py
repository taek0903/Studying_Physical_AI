import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import torch.nn.functional as F
from PIL import Image
from collections import OrderedDict
torch.manual_seed(0)

class GramMatrix(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        # 입력 텐서 batch, channel, weight, width 추출
        b, c, h, w = input.size()

        # 텐서를 (batch, channel, height * width) 형태로 flatten(평탄화)함
        # 이 구현에서 batch 차원은 유지하고 channel 차원만 평탄화 해야함 (b,c,h*w)
        F = input.view(b,c,h*w)

        '''
        배치 행렬 곱셈(bmm) 사용, gram matrix (G) 계산
        F(feature map: 특징맵)와 F의 채널-공간 전치(transpose)를 곱합, 결과 크기(b,c,c)
        원래 f.shape = (b,c,h*w) h*w를 n이라고 한다면 (b,c,n)
        batch는 남겨두고 F.transpose(1,2)하면 (b,n,c)
        두 features map간 내적하면 (b,c,c)
        의미 : batch_size는 남겨주고 c,c 채널 간 상관관계 파악
        i번쨰 채널과 j번쨰 채널이 얼마나 이미지에 동시에 활성화되었는가 파악
        w*h=n이 였는데 없어짐 => 위치정보 제거됨
        남은 것은 스타일(질감, 색상)만 남음
        '''

        G = torch.bmm(F, F.transpose(1,2))

        # Gram matrix을 height * width 나눔 => 정규화
        G.div_(h*w)

        return G
    
# Gram Matrix 이용, MSE(평균 제곱 오차) 손실을 계산하는 모듈 정의
class GramMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input, target):
        # GramMatrix 모듈 통과 gram matrix 계산함
        # => 목표(target) gram matrix MSE 손실계산
        out = nn.MSELoss()(GramMatrix()(input), target)
        # 계산된 스타일 손실 변환
        return (out)
    # nn.MSELoss() => 클래스 생성 (GramMatrix()생성 후 input변수를 넣어 실행)
    # => GramMatrix 클래스에 input을 넣어 실행한후 target과 함께 nn.MseLoss 클래스 실행

class VGG(nn.Module):
    def __init__(self, pool='max'):
        super().__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        if pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool3(out['r54'])
        return [out[key] for key in out_keys]
    # 특정 layer 이름 지정 (r11, r42) => 결과값만 골라오기 위해서

vgg = VGG()

img1 = r'D:\rokey\AI_applications\data\vangogh_starry_night.jpg'
img2 = r'D:\rokey\AI_applications\data\Tuebingen_Neckarfront.jpg'

# 이미지 디렉토리와 파일 이름을 결합하여 PIL 객체 리스트로 로드함
img1 = Image.open(img1)
img2 = Image.open(img2)
imgs = []
imgs.append(img1)
imgs.append(img2)

img_size = 512

prep = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x : x[torch.LongTensor([2,1,0])]),  # RGB => BGR 변환
    transforms.Normalize(mean=[0.407,0.457,0.485], std=[1,1,1]),
    transforms.Lambda(lambda x: x.mul_(255))
])

img_torch = [prep(img) for img in imgs]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

imgs_torch = [img.unsqueeze(0).to(device) for img in img_torch]

# 이미지 텐서 리스트를 스타일 이미지와 콘텐츠 이미지 변수에 할당
style_image, content_image = imgs_torch

opt_img = content_image.detach().clone().requires_grad_(True)
# detach() 자동미분 계산하면서 이전 계산 기록을 끊어내기 위해 사용
# requires_grad_(True) 모델의 가중치 변경하는게 아님 이미지가 가진 pixel 학습대상

vgg=vgg.to(device)

'''
스타일 손실 계산항 VGG layer 이름 정의ㅏ
r11, r21 => vgg 내부의 특징 합성곱 레이어 나타냄(relu 통과한 층)
낮은 레이어 (이미지의 아주 세밀한 형태 경계선, 점, 색상)
'''
style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']

'''
콘텐츠 손실 계산할 vgg layer 이름 정의
일반적으로 중간 레이어 하나사용
=> 객체의 눈, 코, 입 전체적인 윤곽이 나옴
'''

content_layers = ['r42']

loss_layers = style_layers + content_layers
# 사용할 모든 손실 레이어 = 스타일 손실 + 콘텐츠 손실

# 각 스타일에 대해서 GramMSELoss 모듈 적용
# 콘텐츠에 대해서는 MSELoss 모듈 사용

loss_fns = [GramMSELoss().to(device) for _ in style_layers] +\
           [nn.MSELoss().to(device) for _ in content_layers]
'''
스타일 손실에 부여할 가중치 정의함
깊은 레이어일 수록 낮은 가중치를 주는 경향이 있음
최종적으로 스타일 손실은 (b,c,c) c*c
채널수가 적은 쪽에 낮은 가중치. 채널수가 많은 뒤쪽에는 작은 가중치를 부여함
전체적인 손실 규모를 맞춰주게 됨 => 일종의 정규화 효과
=> 깊은 층일수록 연산되는 채널 수가 많아져
Gram Matrix(스타일 특징값)의 스케일이 비정상적으로 커지게 되고
이로 인한 과도한 오차(Loss)를 억제하여 층별 균형을 맞추기 위해 가중치를 작게 설정한다.
'''

style_weights = [1e3/n**2 for n in [64, 128, 256, 512, 512]]

# 콘텐츠 손실에 부여할 가중치(weights) 정의
content_weights = [1e0]

# 최종적으로 사용할 모든 가중치 리스트
weights = style_weights + content_weights

# 최적화 목표값(style trargets) 계산
# style_image를 vgg에 통과 시켜 각 style_layers의 Gram Matrix 계산
style_targets = \
[GramMatrix()(A).detach() for A in vgg(style_image, style_layers)]

# 최적화 목표값(content targets)
content_targets = \
[A.detach() for A in vgg(content_image, content_layers)]

# 최종적으로 사용할 모든 목표값 리스트 정의
targets = style_targets + content_targets

input_image = content_image.clone().requires_grad_(True)
# input_image는 vgg에 입력될 초기 이미지. 곤첸츠 이미지와 동일한 형태여야만 함
# reqires_grad_(True)의 의미
# 입력 이미지의 픽셀 값을 어떻게 변화시켜야 스타일을 입힐 수 있는지 계산하기위한
# 이미지에 대한 기울기(gradient) 추적을 켜는 것
optimizer = torch.optim.LBFGS([input_image], max_iter=1)

'''
LBFGS 특징 : 2차미분(헤시안hassian) 근사치 사용(극소값, 극대값, 안정값 판별)
왜 Adam 보다 선호되는가? 2차 해시안 사용하여 근사치를 구한 결과가 더 정교하고 빠르게 최적의
픽셀을 찾아내는 데 도움을 준다.
LBFGS는 적은 데이터에 더 선명하고 고품질의 결과물을 가져오는 경향이 있음
LBFGS 사용하면 closure 사용해야함 (최대 단점: 현재 상태에서 여러번 기울기 계산)
'''

# 최적화 횟수를 세기 위한 변수
n_iter = 0

def closure():
    global n_iter

    optimizer.zero_grad()

    # 생성된 이미지(input_image)를 vgg 통과 => 특징맵 추출
    out = vgg(input_image, loss_layers)

    # 총손실 계산
    layer_losses = []
    total_loss = 0

    for i, weight in enumerate(weights):
        target = targets[i]
        feature = out[i]
        loss_fn = loss_fns[i]

        loss = weight * loss_fn(feature, target)
        layer_losses.append(loss.item())
        total_loss += loss

    # 역전파 수행
    total_loss.backward()

    # 진행상황 출력
    if n_iter % 50 == 0:
        print(f'Iteration {n_iter}: Total Loss={total_loss.item():.4f}')

    n_iter +=1
    return total_loss

# 최적화 실행(반복 횟수 지정)
num_iterations = 500

for i in range(num_iterations):
    optimizer.step(closure)
    # LBSFGS 는 step() 호출 할 때마다 closure() 여러 번 호출 가능

# 최종 결과 이미지 후처리
# 생성된 이미지를 [0,1] 범위로 클리핑하여 픽셀값 보정
input_image.data.clamp_(0,1)

def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)    # 배치차원 제거 (1, C, H, W) => (C, H, W)
    unloader = transforms.ToPILImage()  # PIL이미지로 변경
    image = unloader(image)

    if title is not None:
        plt.title(title)
    plt
    plt.pause(0.001)    # 잠시 멈춰서 그림 보여줘

plt.figure(figsize=(10,10))
plt.imshow(input_image.data.cpu().squeeze(0).permute(1, 2, 0))
plt.title('Finalized_Image')
plt.show()