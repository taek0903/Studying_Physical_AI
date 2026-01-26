import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torchviz import make_dot
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
plt.rcParams["font.family"] = "Malgun Gothic"

from pythonlibs.torch_lib1 import *

# 적응형 풀링 함수(nn.AdaptiveAvgPool2d 함수)
p = nn.AdaptiveAvgPool2d((1,1))     # 출력을 (1, 1) 크기로 만드는 폴링 레이어 정의

l1 = nn.Linear(32, 10)  # 입력 32, 출력 10의 선형 레이어 정의

# 더미 데이터 생성 (배치크기=100, 채널=32, 높이=16, 너비=16)
inputs = torch.randn(100, 32, 16, 16)

m1 = p(inputs)                  # (100, 32, 16, 16) -> (100, 32, 1, 1)
print(m1.shape[0])              # 100
m2 = m1.view(m1.shape[0], -1)   # 선형 레이어 입력을 위해 1차원으로 펼침(Flatten) => (100, 32)
m3 = l1(m2)                     # 최종 예측 결과 -> (100, 10)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

n_output = len(classes)
# 학습 데이터용 : 리사이즈, 좌우반전, 텐서 변환, 정규화, 랜덤 지우기 적용
transform_train = transforms.Compose([
    transforms.Resize(112),              # 이미지 크기를 112X112로 조정
    # 50% 확률로 이미지 좌우 뒤집기(사물이 반대방향 보고 있어도 동일한 객체인지 학습시키기 위함)
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),    
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 3채널 이미지에 대한 정규화 RGB
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
    # p=0.5 = 50%
    # scale=(0.02, 0.33) 전체 이미지 면적에서 지워지는 영역이 차지하는 비율 범위 2%~33%
    # ratio=(0.3, 3.3) 지워지는 사각형 영역의 세로가로 비율 (세로 0.3, 가로 3.3) => w/h = ratio
    # value=0 지워지는 영역에 채워지는 값
    # inplace=False: 원본 이미지는 그대로 두고 지원 결과를 새 텐서로 만들어 반환
    # inplace=True: 입력으로 들어온 이미지 텐서를 직접 수정, 원본이 바뀌어 디버깅이 어려울 수도 있음
])

# 검증 데이터용 : 리사이즈, 텐서변환, 정규화'만' 사용
transform_test = transforms.Compose([
    transforms.Resize(112),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

data_root = './data'

# 학습 데이터 셋
train_set = datasets.CIFAR10(
    root=data_root, train=True,
    download=True, transform=transform_train
)

# 검증 데이터 셋(train=False 설정하기)
test_set = datasets.CIFAR10(
    root=data_root, train=False,
    download=True, transform=transform_test
)

# 배치 사이즈 지정
batch_size = 50

# 데이터 로더

# 훈련용 데이터로더(shuffle=True)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# 검증용 데이터로더(shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

from torchvision import models

# # 사전 학습 모델 불러오기
# # pretrained = True로 학습을 마친 파리미터를 동시에 불러오기
# net = models.resnet(pretrained = True)

# torch_seed()

# # 최종 레이어 함수의 입력 차원수 확인 (e.g., ResNet-18의 경우 512)
# fc_in_features = net.fc.in_features     # 기존 입력 차원 512

# # 최종 레이어 함수를 새로운 nn.Linear로 교체
# # 입력은 그대로, 출력은 우리의 클래스 개수(n_output)로 설정
# net.fc = nn.Linear(fc_in_features, n_output)    # n_output = 10

# 사전 학습 모델 불러오기
# pretrained = True로 학습을 마친 파라미터도 함께 불러오기
net = models.resnet18(pretrained = True)

# 최종 레이어 함수 입력 차원수 확인
fc_in_features = net.fc.in_features

# 최종 레이어 함수 교체
net.fc = nn.Linear(fc_in_features, n_output)

net = net.to(device)

lr = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
history = np.zeros((0,5))
num_epochs = 5
history = fit(net, optimizer, criterion, num_epochs,
              train_loader, test_loader, device, history)

# 결과 요약하기
evaluate_history(history)

# 이미지와 정답, 예측 결과를 함께 표시
show_images_labels(test_loader, classes, net, device)