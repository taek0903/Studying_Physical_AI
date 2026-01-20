import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import matplotlib.font_manager as fm
import torch
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot

device = 'cuda' if torch.cuda.is_available() else 'cpu'

plt.rcParams["font.family"] = "Malgun Gothic"

print('예측 함수의 내부 구조')
# 코드 4-1 : 레이어 함수 정의
# 레이어 함수 인스턴스 생성
l1 = nn.Linear(784, 128)    # 첫번째 선형 계수 784개 입력 받아 128개로 출력
l2 = nn.Linear(128, 10)     # 두번째 선형 계수 128개 입력받아 10개(0-9 숫자)로 출력
relu = nn.ReLU(inplace=True)    # 활성화 함수

# 코드 4-2: 예측 함수의 간략한 구현 예시
# 더미 입력 데이터 생성
inputs = torch.randn(100, 784)  # 100개의 데이터, 각 데이터는 784개의 요소를 가짐

# 입력 텐서로부터 출력 텐서 계산
m1 = l1(inputs) # 1. 첫 번쨰 선형 함수 통과
m2 = relu(m1)   # 2. 활성화 함수 통과
outputs = l2(m2)    # 3. 두번째 선형 함수 통과

# 입력과 출력 텐서의 shape 확인
print(f'입력 텐서 shape: {inputs.shape}')
print(f'출력 텐서 shape: {outputs.shape}')

# 코드 4-3: nn.Sequential의 사용 예시
# nn.Sequential을 사용해 전체를 하나의 합성 함수로 정의
net2 = nn.Sequential(
    l1,
    relu,
    l2
)

# 한 번에 결과 계산
outputs2 = net2(inputs)

# 입력 텐서와 출력 텐서의 shape 확인
print(f'입력 텐서 shape: {inputs.shape}')
print(f'출력 텐서 shape: {outputs.shape}')

# 코드 4-4 학습용 데이터를 생성하는 프로그램
np.random.seed(123)     # 재현성을 위해 난수 시드 고정

x= np.random.randn(100,1) * 2.5 # x 데이터 생성 (-2.5 ~ 2.5 사이의 정규분포 따르는 난수 100개)

y = x**2 + np.random.randn(100,1) * 0.8 # y는 x^2에 난수를 더한 값

# 데이터를 50건씩 훈련용과 검즘용으로 나눔
x_train = x[:50, :]
y_train = y[:50, :]
x_test = x[50:, :]
y_test = y[50:, :]


# 코드 4-5 학습 데이터의 산포도 출력
# 산포도 출력
plt.scatter(x_train, y_train, c='c', label='훈련 데이터')
plt.scatter(x_test, y_test, c='k', marker='x', label='검증 데이터')
plt.legend()
# plt.show()

# 입력 변수 x와 정답 yt의 텐서화
inputs = torch.tensor(x_train).float()  # 입력 데이터
labels = torch.tensor(y_train).float()  # 정답 데이터

inputs_test = torch.tensor(x_test).float()
labels_test = torch.tensor(y_test).float()

class Net(nn.Module):
    def __init__(self):
        super().__init__()    # 부모 클래스 nn.Module 의 초기화
        self.l1 = nn.Linear(1, 1)  # 출력층 정의
    
    # 예측 함수 정의
    def forward(self, x):
        x1 = self.l1(x) # 선형 회귀
        return x1

lr = 0.01   # 학습률

net = Net() # 인스턴스 생성(파라미터 초기화)
optimizer = optim.SGD(net.parameters(), lr = lr)    # 최적화 알고리즘 : 경사 하강법
criterion = nn.MSELoss()    # 손실 함수 : 평균 제곱 오차
num_epochs = 1000 # 반복 횟수
history = np.zeros((0,2))   # history 기록을 위한 배열 초기화

# 반복 계산 메인 루프
for epoch in range(num_epochs):
    optimizer.zero_grad()   # 경사 값 초기화
    outputs = net(inputs)   # 예측 계산
    loss = criterion(outputs, labels)   # 경사 계산
    loss.backward()
    optimizer.step()    # 경사 하강법 적용
    if (epoch % 100 == 0):
        history = np.vstack((history, np.array([epoch, loss.item()])))
        print(f'Epoch {epoch} loss: {loss.item():.5f}')

print('유사 딥러닝 모델의 경우')
class Net2(nn.Module):
    def __init__(self):
        super().__init__()

        # 출력층 정의
        self.l1 = nn.Linear(1, 10)
        self.l2 = nn.Linear(10, 10)
        self.l3 = nn.Linear(10, 1)

    # 예측 함수 정의
    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        return x3
lr = 0.01

net2 = Net2()   # 인스턴스 생성(파라미터 초기화)
optimizer = optim.SGD(net.parameters(), lr=lr)  # 최적화 알고리즘 : 경사 하강법
criterion = nn.MSELoss()    # 손실 함수 : 평균 제곱 오차
num_epochs = 1000   # 반복 횟수
history = np.zeros((0,2))   # history 기록을 위한 배열 초기화(손실 함수 값 만을 기록)

# 반복 계산 메인 루프
for epoch in range(num_epochs):
    optimizer.zero_grad()   # 경사 값 초기화
    outputs = net2(inputs)  # 예측 계산
    loss = criterion(outputs, labels)   # 오차 계산
    loss.backward() # 경사 계산
    optimizer.step()    # 경사 하강법 적용

if (epoch % 100 == 0):
    history = np.vstack(history, np.array([epoch, loss.item()]))
    print(f'Epoch {epoch} loss: {loss.item():.5f}')

labels_pred2 = net2(inputs_test)

plt.title('은닉층 2개, 활성화 함수 사용하지 않음')
plt.scatter(inputs_test[:,0].data, labels_pred2[:,0].data, c='b', label='예측값')
plt.scatter(inputs_test[:,0].data, labels_test[:,0].data, c='k', marker='x',label='정답')
plt.legend()
# plt.show()

print('딥러닝 모델(활성화 함수 사용)의 경우')
class Net3(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1,10)
        self.l2 = nn.Linear(10,10)
        self.l3 = nn.Linear(10,1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.l1(x))
        x2 = self.relu(self.l2(x1))
        x3 = self.relu(self.l3(x2))
        return x3
    
lr = 0.01

net3 = Net3()

optimizer = optim.SGD(net3.parameters(), lr=lr)
criterion = nn.MSELoss()
num_epochs = 1000

for epochs in range(num_epochs):
    optimizer.zero_grad()
    outputs = net3(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    if (epoch % 100 == 0):
        history = np.vstack(history, np.array([epoch, loss.item()]))
        print(f'Epoch {epoch} loss: {loss.item():5f}')

# 결과 그래프
labels_pred3 = net3(inputs_test)

plt.title('은닉층 2개, 활성화 함수 사용')
plt.scatter(inputs_test[:,0].data, labels_pred3[:,0].data, c='b', label='예측값')
plt.scatter(inputs_test[:,0].data, labels_test[:,0].data, c='k', marker='x',label='정답')
plt.legend()
# plt.show()