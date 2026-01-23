import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchinfo import summary
from torchviz import make_dot
from tqdm import tqdm
import torchvision.datasets as datasets
from sklearn.datasets import load_digits  # 손글씨 숫자 데이터 (0~9)
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

plt.rcParams["font.family"] = "Malgun Gothic"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 분류 클래스 명칭 리스트
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 분류 클래스 수, 10
n_output = len(list(set(classes)))

# 데이터 준비
from pythonlibs.torch_lib1 import *
print(README)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5) #(평균 mean, 표준편차 std) 범위 [-1.1], 편차 0.5
])

data_root = './data'

train_set = datasets.CIFAR10(
    root = data_root, train = True,
    download = True, transform=transform)

# 검증 데이터 셋
test_set = datasets.CIFAR10(
    root = data_root, train = False,
    download=True, transform=transform)

# 미니 배치 사이즈 지정
batch_size = 100

# 훈련용 데이터로더
# 훈련용 => 셔플 True 설정
train_loader = DataLoader(train_set,
                          batch_size, shuffle=True)

# 검증용 데이터로더
# 검증용 => 셔플 Flase 설정
test_loader = DataLoader(test_set,
                         batch_size, shuffle=False)

show_images_labels(test_loader, classes, None, None) # None, None: 모델, 디바이스

class CNN_v2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 합성곱, ReLU, MaxPool 레이어 정의
        self.conv1 = nn.Conv2d(3, 32, 3, padding=(1,1)) # padding=(1,1) 이미지 크기 유지
        self.conv2 = nn.Conv2d(32, 32, 3, padding=(1,1))
        self.conv3 = nn.Conv2d(32, 64, 3, padding=(1,1))
        self.conv4 = nn.Conv2d(64, 64, 3, padding=(1,1))
        self.conv5 = nn.Conv2d(64, 128, 3, padding=(1,1))
        self.conv6 = nn.Conv2d(128, 128, 3, padding=(1,1))
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.maxpool = nn.MaxPool2d((2,2))    # (3, 28, 28) => (3, 14, 14)절반 줄여줌

        # 선형(Fully Connected) 레이어 정의
        self.l1 = nn.Linear(4*4*128, 128)
        self.l2 = nn.Linear(128, num_classes)

        # 레이어를 순차적으로 실행할 'nn.Sequential' 정의
        # 특성 추출 레이어
        self.features = nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.maxpool,
            self.conv3,
            self.relu,
            self.conv4,
            self.relu,
            self.maxpool,
            self.conv5,
            self.relu,
            self.conv6,
            self.relu,
            self.maxpool
        )

        # 선형 레이어 순차적으로 실행할 'nn.Sequential'정의
        self.classifier = nn.Sequential(
            self.l1,
            self.relu,
            self.l2
        )

    def forward(self, x):
        x1 = self.features(x)
        x2 = self.flatten(x1)   # 2차원 데이터 1차원 평탄화
        x3 = self.classifier(x2)
        return x3
# 난수 고정
torch_seed()

# 모델 인스턴스 생성
lr = 0.01
net = CNN_v2(n_output).to(device)
criterion = nn.CrossEntropyLoss()
loss = eval_loss(test_loader, device, net, criterion)
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
# momentum=0.9 : 이전 업데이트되는 방향을 90% 반영 => 학습속도 높이고 안정화
history = np.zeros((0,5))
num_epochs = 50

g = make_dot(loss, params=dict(net.named_parameters()))
display(g)

history = fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history)
evaluate_history(history)