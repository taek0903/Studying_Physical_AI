import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F # 함수 신경망 연산(F) 모듈을 불러옴, 활성화 함수나 풀링 등에 사용
import torchvision.transforms as transforms
torch.manual_seed(55)

# LeNet 클래스를 정의함 nn.Module을 상속받아 PyTorch 신경망 모듈로 만듦
class LeNet(nn.Module):
    # 클래스의 인스턴스를 초기화 함. 레이어들을 정의함
    def __init__(self):
        super().__init__()

        # 첫 번째 합성곱 레이어(cn1)를 정의함.
        # 입력 채널 3개(RGB 이미지), 출력 특징 맵 6개, 커널 크기 5X5를 사용함
        self.cn1 = nn.Conv2d(3, 6, 5)   # 1층 rgb (3) => 6개 특징 (직선, 곡선 저수준)

        # 두 번쨰 합성곱 레이어(cn2)를 정의함.
        # 입력 채널 6개 출력 특징 맵 16개, 커널 크기 5X5를 사용함
        self.cn2 = nn.Conv2d(6, 16, 5)  # 2층 6개 => 16개 특징 (형태 등 중-고수준)  // 여기까지 특징

        # 분류기(classifier)
        # 첫 번째 완전연결 레이어(fc1)를 정의함
        # 입력 크기는 16 * 5 * 5 (이전)
        self.fc1 = nn.Linear(16*5*5, 120)   # 3층 400개 => 120개

        # 두 번쨰 완전 연결 레이어 (fc2)를 정의함.
        # 입력 120. 출력 84를 사용함.
        self.fc2 = nn.Linear(120, 84)   # 4층 120개 => 84개

        # 세 번째 완전 연결 레이어 (fc3)를 정의함
        # 입력 크기는 84, 최종 출력 클래스 수 10개를 사용함
        self.fc3 = nn.Linear(84, 10)    # 5층 84개 => 10개 클래스

    def forward(self, x):

        # cn1을 적용하고 ReLU 활성화 함수를 통과시킴
        x = F.relu(self.cn1(x))

        # 2x2 크기의 맥스 풀링을 적용함. 공간 크기를 절반으로 줄임.
        x = F.max_pool2d(x, (2, 2))  
        
        # cn2를 적용하고 ReLU 활성화 함수를 통과시킴.
        x = F.relu(self.cn2(x))

        # 2X2 크기의 맥스 풀링을 다시 적용함.
        x = F.max_pool2d(x, (2,2))

        '''
        데이터를 평탄화(flatten)함. 배치 차원(-1)을 제외한 모든 차원을 하나의 벡터로 만듦.
        view() 에서 -1 은 batch_size 자동 계산
        self.flattened_features(x) h*w (이미지) * channel(1: 흑백, 3: 컬러)
        => batch dimension 유지, 나머지 차원 (h*w*c) >> 1차원 벡터로 변환
        Before flatten) batch_size = 32, x.shape = [32, 16, 5, 5] c=16, h=5, w=5
        => flatten) x.view(-1, 16*5*5) = (-1, 400) >> x.shape = [32, 400]        
        '''
        x= x.view(-1, self.flattened_features(x))

        # 첫 번째 완전 연결 레이어와 ReLU를 통과시킴
        x = F.relu(self.fc1(x))

        # 두 번째 완전 연결 레이어와 ReLU를 통과시킴
        x = F.relu(self.fc2(x))

        # 최종 출력 레이어를 통과시킴
        x = self.fc3(x)

        # 최종 결과를 반환함
        return x
    # 데이터를 평탄화하기 위해 특징들의 총 개수를 계산하는 헬퍼 함수임
    def flattened_features(self, x):
        # 배치 차원(첫 번째 차원)을 제외한 나머지 차원들의 크기를 가져옴
        size = x.size()[1:]
        num_feats = 1

        # 모든 차원 크기를 곱하여 총 특징 개수를 계산함.
        for s in size:
            num_feats *= s

        # 총 특징 개수를 반환함.
        return num_feats

lene = LeNet

# print(lenet)
'''
LeNet 구조
1. Convolution(합성곱) + 활성화(ReLU/tanh)
 - 작은 필터로 이미지를 훑으면서 모서리, 선, 질감 같은 특징을 뽑음.
2. Pooling(풀링)
 - 특징맵을 조금 줄여서 계산량을 줄이고, 위치 변화에 덜 민감하게 만듦.
3. Fully Connected(완전연결층)
 - 마지막에 뽑아낸 특징들을 모아서 클래스(숫자/글자 등)로 분류

데이터가 작고 단순하거나 빠르게 베이스 라인을 만들거나 
임베디드 저사양에서 아주 간단한 분류에서 사용하면 좋다.
'''