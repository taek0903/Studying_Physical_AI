import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torchviz import make_dot
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

plt.rcParams["font.family"] = "Malgun Gothic"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 데이터 불러오기
iris = load_iris()

# 입력 데이터와 정답 데이터
x_org, y_org = iris.data, iris.target

# 결과 확인
print('원본 데이터', x_org.shape, y_org.shape)

# 데이터 추출
# 입력 데이터로 sepal length와 petal length 추출
x_select = x_org[:, [0,2]]

# 결과 확인
print('원본 데이터', x_select.shape, y_org.shape)

# 훈련 데이터와 검증 데이터로 분할(셔플도 동시에 실시함)
x_train, x_test, y_train, y_test = train_test_split(
    x_select, y_org, train_size=75, test_size=75,
    random_state=42)

print('--- 데이터 분할 확인 ---')
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# # 산포도 출력
# x_t0 = x_train[y_train == 0]
# x_t1 = x_train[y_train == 1]
# x_t2 = x_train[y_train == 2]

# # 실데이터 분포 확인 > 어떤 변수끼리 상관관계가 있나 확인
# plt.scatter(x_t0[:,0], x_t0[:,1], marker='x', c='k', s=50, label='0 (setosa)')
# plt.scatter(x_t1[:,0], x_t1[:,1], marker='o', c='b', s=50, label='1 (versicolor)')
# plt.scatter(x_t2[:,0], x_t2[:,1], marker='+', c='k', s=50, label='2 (virginica)')
# plt.xlabel('sepal_length')
# plt.ylabel('petal_length')
# plt.legend()
# plt.show()

# 학습용 파라미터 설정

# 입력 차원수
n_input = x_train.shape[1]  # x_train의 클래스(속성) 개수 2개

# 출력 차원수
n_output = len(list(set(y_train)))  # 분류 클래스 수, 3 y_train의 중복제외 클래스 [0,1,2]

print('--- 입출력 차원 확인 ---')
print(f'n_input: {n_input}  n_output: {n_output}')

# 모델 정의
# 2입력 3출력 로지스틱 회귀 모델

class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.l1 = nn.Linear(n_input, n_output)

        # 초깃값을 모두 1로 함
        # "딥러닝을 위한 수학"과 조건을 맞추기 위한 목적
        self.l1.weight.data.fill_(1.0)
        self.l1.bias.data.fill_(1.0)

    def forward(self, x):
        return self.l1(x)

net = Net(n_input, n_output).to(device)

print('--- 모델 파라미터 확인 ---')
for parameter in net.named_parameters():
    print(parameter)

# 모델 개요 표시
print(net)
summary(net, (2,))

# 입력 데이터 x_train과 정답 데이터 y_train의 텐서 변수화
inputs = torch.tensor(x_train).float().to(device)
labels = torch.tensor(y_train).long().to(device)

# 검증 데이터의 텐서 변수화
inputs_test = torch.tensor(x_test).float().to(device)
labels_test = torch.tensor(y_test).long().to(device)

# 인스턴스 생성 
net = Net(n_input, n_output).to(device)
lr = 0.01   # 학습률
criterion = nn.CrossEntropyLoss()   # 손실함수: 교차 엔트로피 함수(softmax()+log+정답뽑기)
optimizer = optim.SGD(net.parameters(), lr=lr)
history = np.zeros((0,5))
num_epochs = 10000

for epoch in range(num_epochs):
    # 훈련 세션
    optimizer.zero_grad()   # 경사 초기화
    outputs = net(inputs)   # 예측 계산
    loss = criterion(outputs, labels)   # 손실 계산
    loss.backward()     # 경사 계산
    optimizer.step()    # 파라미터 수정
    predicted = torch.max(outputs, 1)[1]    # 예측 라벨 산출
    train_loss = loss.item()    # 손실 값을 파이썬에 사용할 수 있는 데이터 값으로 추출
    train_acc = (predicted == labels).sum() / len(labels)   # 정확도 계산

    # 예측 세션
    outputs_test = net(inputs_test) # 테스트 예측 계산
    loss_test = criterion(outputs_test, labels_test)    # 테스트 손실 계산
    predicted_test = torch.max(outputs_test, 1)[1]      # 테스트 예측 라벨 산출
    val_loss = loss_test.item()     # 테스트 손실값 추출 
    val_acc = (predicted_test == labels_test).sum() / len(labels_test)   # 테스트 정확도 계산

    if ((epoch) % 1000 == 0):
        print (f'Epoch [{epoch}/{num_epochs}], loss: {train_loss:.5f} acc: {train_acc:.5f} val_loss: {val_loss:.5f}, val_acc: {val_acc:.5f}')
        item = np.array([epoch, float(train_loss), float(train_acc), 
                         float(val_loss), float(val_acc)], dtype=np.float32)
        history = np.vstack((history, item))

print(torch.max(outputs, 1))        # 2번째 인수는 축을 의미 1이면 행별로 집계
print(torch.max(outputs, 1)[0])     # torch.max 함수의 행렬에 value값을 추출
print()
print(torch.max(outputs, 1)[1])     # torch.max 함수의 행렬에 index값을 추출

# 손실과 정확도 확인
print(f'초기상태 : 손실 : {history[0,3]:.5f}  정확도 : {history[0,4]:.5f}' )
print(f'최종상태 : 손실 : {history[-1,3]:.5f}  정확도 : {history[-1,4]:.5f}' )

# # 학습 곡선 출력(손실)
# plt.plot(history[:,0], history[:,1], 'b', label='훈련')
# plt.plot(history[:,0], history[:,3], 'k', label='검증')
# plt.xlabel('반복 횟수')
# plt.ylabel('손실')
# plt.title('학습 곡선(손실)')
# plt.legend()
# plt.show()

# # 학습 곡선 출력(정확도)
# plt.plot(history[:,0], history[:,2], 'b', label='훈련')
# plt.plot(history[:,0], history[:,4], 'k', label='검증')
# plt.xlabel('반복 횟수')
# plt.ylabel('정확도')
# plt.title('학습 곡선(정확도)')
# plt.legend()
# plt.show()

# 정답 데이터 0, 1, 2에 해당하는 샘플 각각 추출
print('--- 모델 출력값 확인 ---')
indices_to_check = [0, 2, 3]
print(f'확인할 샘플의 정답 레이블: {labels[indices_to_check].cpu().numpy()}')

# 해당 입력값 추출
i3 = inputs[indices_to_check, :]
print(f'해당 입력값:\n{i3.data.cpu().numpy()}')

# 소프트맥스 함수 정의 및 적용
# 모델의 logits를 합이 1인 확률 분포로 바꾸는 함수
softmax = torch.nn.Softmax(dim=1)   # 열방향
o3 = net(i3)
k3 = softmax(o3)

print(f'\n모델의 원시 출력 (logits):\n{o3.data.cpu().numpy()}')
print(f'소프트맥스 적용 후 확률:\n{k3.data.cpu().numpy()}')

# 가중치 행렬
print(f'가중치 행렬:\n{net.l1.weight.data}')
# 편향
print(f'편향:\n{net.l1.bias.data}')

# 훈련 데이터와 검증 데이터로 분할(셔플도 동시에 실시)
x_train, x_test, y_train, y_test = train_test_split(
    x_org, y_org, train_size=75, test_size=75,
    random_state=123)   
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# 입력 차원수
n_input = x_train.shape[1]
print(n_input)

inputs = torch.tensor(x_train).float().to(device)
labels = torch.tensor(y_train).long().to(device)
inputs_test = torch.tensor(x_test).float().to(device)
labels_test = torch.tensor(y_test).long().to(device)

lr = 0.01
net = Net(n_input, n_output).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr)
num_epochs = 10000
history = np.zeros((0,5))

for epoch in range(num_epochs):
    # 훈련세션
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    predicted = torch.max(outputs, 1)[1]
    train_loss = loss.item()
    train_acc = (predicted == labels).sum() / len(labels)

    # 예측세션
    outputs_test = net(inputs_test)
    loss_test = criterion(outputs_test, labels_test)
    predicted_test = torch.max(outputs_test, 1)[1]
    val_loss = loss.item()
    val_acc = (predicted_test == labels_test).sum() / len(labels_test)
    
    if ((epoch) % 1000 == 0):
        print(f"Epoch [{epoch}/{num_epochs}], "
                f"loss: {train_loss:.5f} acc: {train_acc:.5f} "
                f"val_loss: {val_loss:.5f} val_acc: {val_acc:.5f}")
        item = np.array([epoch, float(train_loss), float(train_acc), 
                         float(val_loss), float(val_acc)], dtype=np.float32)
        history = np.vstack((history, item))

# 손실과 정확도 확인
print(f'초기상태 : 손실 : {history[0,3]:.5f}  정확도 : {history[0,4]:.5f}' )
print(f'최종상태 : 손실 : {history[-1,3]:.5f}  정확도 : {history[-1,4]:.5f}' )

# # 학습 곡선 출력(손실)
# plt.plot(history[:,0], history[:,1], 'b', label='훈련')
# plt.plot(history[:,0], history[:,3], 'k', label='검증')
# plt.xlabel('반복 횟수')
# plt.ylabel('손실')
# plt.title('학습 곡선(손실)')
# plt.legend()
# plt.show()

# # 학습 곡선 출력(정확도)
# plt.plot(history[:,0], history[:,2], 'b', label='훈련')
# plt.plot(history[:,0], history[:,4], 'k', label='검증')
# plt.xlabel('반복 횟수')
# plt.ylabel('정확도')
# plt.title('학습 곡선(정확도)')
# plt.legend()
# plt.show()

'''
2채널 입력값은 train_acc가 test_acc보다 작은 것을 확인 할 수 있다. 
이는 과소적합이 되었다고 분석을 할 수 있다.
즉, 학습 데이터 조차 잘 설명을 못하고 테스트 데이터가 운좋게 더 맞았다고도 할 수 있다.
4채널 입력값에서는 train_acc가 test_acc값 보다 커지는 것을 확인 할 수 있는데
서로 다른 관점의 정보가 추가 되어 모델이 클래스 간 분리를 하는 데 큰 도움이 되는
새로운 데이터들을 받았다고 생각할 수 있다.
그래프를 비교해보면 4차원 입력이 더욱 안정적으로 학습하는 것을 확인할 수 있다.
2채널 입력과 4채널 입력을 비교하여 얻을 수 있는 것은 분류에 유효한 정보에 따라
모델의 정확도가 달라지고 입력 채널에 따라 과소적합이 생길 수도 있다는 것이다.
'''