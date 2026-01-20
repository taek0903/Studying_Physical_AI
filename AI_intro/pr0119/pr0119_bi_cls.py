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

x_np = np.arange(-4, 4.1, 0.25)
x = torch.tensor(x_np).float()
y = torch.sigmoid(x)

# plt.title('시그모이드 함수의 그래프')
# plt.plot(x.data, y.data)
# plt.show()

iris = load_iris()
x_org, y_org = iris.data, iris.target

print('--- 1. 데이터 불러오기 결과 ---')
print(f'원본 데이터 {x_org.shape}, {y_org.shape}')  # (150, 4) (150,)

x_data = iris.data[:100,:2]
y_data = iris.target[:100]

print('--- 2. 데이터 추출 결과 ---')
print(f'대상 데이터 {x_data.shape} {y_data.shape}') # (100,2) (100,)

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, train_size=70, test_size = 30, random_state=42
)
print('--- 3. 데이터 분할 결과 ---')
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (70, 2), (30, 2) (70,) (30,)

# x_t0 = x_train[y_train == 0]
# x_t1 = x_train[y_train == 1]
# plt.scatter(x_t0[:,0], x_t0[:,1], marker='x', c='b', label='0 (setosa)')
# plt.scatter(x_t1[:,0], x_t1[:,1], marker='o', c='k', label='1 (versicolor)')
# plt.xlabel('sepal_length')
# plt.ylabel('sepal_width')
# plt.legend()
# plt.show()

n_input = x_train.shape[1]
n_output = 1
print(f'n_inputL {n_input}, n_output: {n_output}')

class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.l1 = nn.Linear(n_input, n_output)
        self.sigmoid = nn.Sigmoid()

        self.l1.weight.data.fill_(1.0)
        self.l1.bias.data.fill_(1.0)

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.sigmoid(x1)
        return x2

net = Net(n_input, n_output).to(device)

for parameter in net.named_parameters():
    print(parameter)
print(summary(net, (2,)))

inputs = torch.tensor(x_train).float().to(device)
labels = torch.tensor(y_train).float().to(device)

labels1 = labels.view((-1,1))

inputs_test = torch.tensor(x_test).float().to(device)
labels_test = torch.tensor(y_test).float().to(device)

labels1_test = labels_test.view((-1,1))

criterion = nn.BCELoss()
lr = 0.01
optimizer = optim.SGD(net.parameters(), lr=lr)

num_epochs = 10000

history = np.zeros((0,5))

for epoch in range(num_epochs):
    # 훈련 페이즈
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels1)
    loss.backward()
    optimizer.step()

    train_loss = loss.item()    # 손실 저장(스칼라 값 취득)

    predicted = torch.where(outputs < 0.5, 0, 1)    # 확률값을 0.5 기준으로 0 또는 1로 변환

    train_acc = (predicted == labels1).sum() / len(y_train)

    # 예측 페이즈
    # 여기서는 경사 계산과 파라미터 수정이 필요 없음
    outputs_test = net(inputs_test)     # 예측 계산
    loss_test = criterion(outputs_test, labels1_test)   # 손실계산
    val_loss = loss_test.item()     # 손실 저장
    predicted_test = torch.where(outputs_test < 0.5, 0, 1)
    val_acc = (predicted_test == labels1_test).sum() / len(y_test)

    if epoch % 1000 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], loss: {train_loss:.5f},' 
              f'acc: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_acc: {val_acc:.5f}')
    
    item = np.array([epoch, float(train_loss), float(train_acc), 
                     float(val_loss), float(val_acc)], dtype=np.float32)
    history = np.vstack((history, item)).astype(np.float32)

print('--- 최종 결과 ---')
print(f'초기 상태(검증) : 손실: {history[0, 3]:.5f}, 정확도: {history[0, 4]:.5f}')
print(f'최종 상태(검증) : 손실: {history[-1, 3]:.5f}, 정확도: {history[-1, 4]:.5f}')   

# plt.plot(history[:,0], history[:,1], 'b', label='훈련')
# plt.plot(history[:,0], history[:,3], 'k', label='검증')
# plt.xlabel('반복 횟수')
# plt.ylabel('손실')
# plt.title('학습 곡선(손실)')
# plt.legend()
# plt.show()

# plt.plot(history[:,0], history[:,2], 'b', label='훈련')
# plt.plot(history[:,0], history[:,4], 'k', label='검증')
# plt.xlabel('반복 횟수')
# plt.ylabel('정확도')
# plt.title('학습 곡선(정확도)')
# plt.legend()
# plt.show()

x_t0 = x_test[y_test == 0]
x_t1 = x_test[y_test == 1]

bias = net.l1.bias.detach().cpu().numpy()
weight = net.l1.weight.detach().cpu().numpy()
print(f'BIAS = {bias}, WEIGHT = {weight}')

def decision(x):
    return(-(bias + weight[0, 0] * x)/ weight[0,1])

xl = np.array([x_test[:,0].min(), x_test[:,0].max()])
yl = decision(xl)

print(f'xl = {xl}  yl = {yl}')

# # 산포도 출력
# plt.scatter(x_t0[:,0], x_t0[:,1], marker='x',
#         c='b', s=50, label='class 0')
# plt.scatter(x_t1[:,0], x_t1[:,1], marker='o',
#         c='k', s=50, label='class 1')

# # 결정 경계 직선
# plt.plot(xl, yl, c='r')
# plt.xlabel('sepal_length')
# plt.ylabel('sepal_width')
# plt.legend()
# plt.show()

class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.l1 = nn.Linear(n_input, n_output)
        self.l1.weight.data.fill_(1.0)
        self.l1.bias.data.fill_(1.0)

    def forward(self, x):
        x1 = self.l1(x)
        return x1

lr = 0.01
net=Net(n_input, n_output).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(net.parameters(), lr=lr)
num_epochs = 10000
history = np.zeros((0,5))

for epoch in range(num_epochs):
    # 훈련 세션
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels1)
    loss.backward()
    optimizer.step()
    train_loss = loss.item()
    # 예측 라벨(1 또는 0) 계산
    predicted = torch.where(outputs < 0.0, 0, 1)
    # 정확도 계산
    train_acc = (predicted == labels1).sum() / len(y_train)

    # 예측 페이즈
    outputs_test = net(inputs_test)
    loss_test = criterion(outputs_test, labels1_test)
    loss_test.backward()
    val_loss = loss_test.item()
    predicted_test = torch.where(outputs_test < 0.0, 0, 1)
    val_acc = (predicted_test == labels_test).sum() / len(y_test)

    if epoch % 1000 == 0:
        print (f'Epoch [{epoch}/{num_epochs}], loss: {train_loss:.5f} acc: {train_acc:.5f}' 
                f'val_loss: {val_loss:.5f}, val_acc: {val_acc:.5f}')
        item = np.array([epoch, float(train_loss), float(train_acc),
                         float(val_loss), float(val_acc)], dtype=np.float32)
        history = np.vstack((history, item)).astype(np.float32)

print(f'초기 상태 : 손실 : {history[0,3]:.5f}  정확도 : {history[0,4]:.5f}' )
print(f'최종 상태 : 손실 : {history[-1,3]:.5f}  정확도 : {history[-1,4]:.5f}' )       

# BCELoss 와 BCEWithLogitsLoss 차이점 BCELoss는 Sigmoid를 포함하지 않기 때문에
# Sigmoid 함수를 모델 설정할 때 넣어줘야 한다. 그리고 판정 자체를 값이 0.5를 기준으로 판별한다.
# BCEWithLogitsLoss는 Sigmoid가 내장 되어 있어서 모델 설정 할 때 Sigmoid함수를 넣지 않아도 된다.
# 그리고 판정을 값이 있냐 없냐로 판정을하기 때문이 0.0 < outputs가 된다.

# 꼭 명심해야 할 것 numpy는 gpu에서 연산할 수 없기에 꼭 cpu로 디바이스 변경을 해준다.
# device 설정을 하면 model과 input, output 같은 변수들에는 .to(device)를 꼭 넣어준다.