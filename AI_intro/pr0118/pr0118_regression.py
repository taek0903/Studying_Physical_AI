import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchviz import make_dot

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

plt.rcParams["font.family"] = "Malgun Gothic"

l = nn.Linear(2,3)
print(l)

torch.manual_seed(123)

l1 = nn.Linear(1,1)

print('l1정보')
print(l1)

print('\nl1의 파라미터')
for param in l1.named_parameters():
    print(f'name: {param[0]}')
    print(f'tensor: {param[1]}')
    print(f'shape: {param[1].shape}')

nn.init.constant_(l1.weight, 2.0)
nn.init.constant_(l1.bias, 1.0)

print(f'wight: {l1.weight}')
print(f'bias: {l1.bias}')

x_np = np.arange(-2.0, 2.1, 1.0)
print(x_np)

x = torch.tensor(x_np).float()
print(x)

x = x.view(-1,1)
print(x)

print(f'x의 shape: {x.shape}')
print(f'x의 값:\n {x}')

y = l1(x)
print(y)

print(f'y의 shape: {y.shape}')
print(f'y의 data: \n {y.data}')

l2 = nn.Linear(2,1)
nn.init.constant_(l2.weight, 1.0)
nn.init.constant_(l2.bias, 2.0)

print("weight:", l2.weight)
print("bias:", l2.bias)

x2_np = np.array([[0,0], [0,1], [1,0], [1,1]])
x2 = torch.tensor(x2_np).float()
print("x2의 shape:", x2.shape)
print("x2의 값:\n", x2)

y2 = l2(x2)

print(y2.shape)

print(y2.data)

l3 = nn.Linear(2,3)

nn.init.constant_(l3.weight[0,:], 1.0)
nn.init.constant_(l3.weight[1,:], 2.0)
nn.init.constant_(l3.weight[2,:], 3.0)
nn.init.constant_(l3.bias, 2.0)

print("weight:", l3.weight)
print("bias:", l3.bias)

y3 = l3(x2)

print("y3의 shape:", y3.shape)
print("y3의 data:\n", y3.data)

class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.l1 = nn.Linear(n_input, n_output)

    def forward(self, x):
        return self.l1(x)
    
inputs = torch.ones(100,1)

n_input = 1
n_output = 1

net = Net(n_input, n_output)

outputs = net(inputs)

print(outputs.shape)
print(outputs[:5])

criterion = nn.MSELoss()
labels = torch.zeros(100,1)
loss = criterion(outputs, labels) / 2.0
loss.backward()
print(f'계산된 손실(loss): {loss.item()}')
print(f'l1.weight.grad: {net.l1.weight.grad}')

outputs = net(inputs)
loss = criterion(outputs, labels) / 2.0
print(f'합성 함수 결과(loss):{loss.item()}')

data_url = "http://lib.stat.cmu.edu/datasets/boston"

raw_df = pd.read_csv(data_url, sep='\s+',
                     skiprows=22, header=None)

x_org = np.hstack([raw_df.values[::2,:],
                   raw_df.values[1::2,:2]])

yt = raw_df.values[1::2, 2]
feature_names = np.array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
                          'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])
print('원본 데이터', x_org.shape, yt.shape)
print('항목명: ', feature_names)

x = x_org[:, feature_names == 'RM']
print('추출 후', x.shape)
print(x[:5, :])

print('정답 데이터')
print(yt[:5])

# plt.scatter(x, yt, s=10, c='b') # x축은 방 개수, y축은 가격
# plt.xlabel('방 개수')
# plt.ylabel('가격')
# plt.title('방 개수와 가격의 산포도')
# plt.grid(True)
# plt.show()

n_input = x.shape[1]
n_output = 1

print(f'입력 차원수: {n_input} 출력 차원수: {n_output}')

class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        # 출력층 정의 : 입력 차원에서 출력 차원으로 가는 선형 변환
        self.l1 = nn.Linear(n_input, n_output)

        # 초깃값을 모두 1로 설정
        # "딥러닝을 위한 수학"과 조건을 맞추기 위함
        nn.init.constant_(l1.weight, 1.0)
        nn.init.constant_(l1.bias, 1.0)

    def forward(self, x):
        x1 = self.l1(x)
        return x1
    
net = Net(n_input, n_output).to(device)

for parameter in net.named_parameters():
    print(f'변수명: {parameter[0]}')
    print(f'변숫값: {parameter[1].data}')

for parameter in net.parameters():
    print(parameter)

print(net)

from torchinfo import summary

summary(net, (1,))

criterion = nn.MSELoss()
lr = 0.01
optimizer = optim.SGD(net.parameters(), lr=lr)

inputs = torch.tensor(x).float().to(device)
labels = torch.tensor(yt).float().to(device)

print(inputs.shape)
print(outputs.shape)

labels1 = labels.view((-1,1))

print(labels.shape)

outputs = net(inputs)
loss = criterion(outputs, labels1)

print(loss)
print(f'loss.item():.5f')

loss.backward()
print(net.l1.weight.grad)
print(net.l1.bias.grad)

optimizer.step()
print(net.l1.weight)
print(net.l1.bias)

optimizer.zero_grad()

lr = 0.01

net=Net(n_input, n_output)
net.to(device)

criterion=nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=lr)

num_epochs = 500
history = np.zeros((0,2))

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels1) / 2.0
    loss.backward()
    optimizer.step()
    if epoch % 100 ==0 :
        history = np.vstack((history, np.array([epoch, loss.item()])))
        print(f'Epoch {epoch} loss: {loss.item():.5f}')
print(history)

print(f'초기 손실값: {history[0,1]:.5f}')
print(f'최종 손실값: {history[-1,1]:.5f}')

# plt.plot(history[1:,0], history[1:,1], 'b')
# plt.xlabel('반복 횟수')
# plt.ylabel('손실')
# plt.title('학습 곡선(손실)')
# plt.show()

xse = np.array((x.min(), x.max())).reshape(-1,1)
Xse = torch.tensor(xse).float().to(device)

with torch.no_grad():
    Yse = net(Xse)

print(Xse.cpu().numpy())

# plt.scatter(x, yt, s=10, c='b')
# plt.xlabel('방 개수')
# plt.ylabel('가격')
# plt.plot(Xse.cpu().data, Yse.cpu().data, c='k')
# plt.title('산포도와 회귀 직선')
# plt.show()

x_add = x_org[:, feature_names == 'LSTAT']
x2 = np.hstack((x,x_add))

print(x2.shape)
print(x2[:5,:])

n_input = x2.shape[1]
print(n_input)

net = Net(n_input, n_output)

for parameter in net.named_parameters():
    print(f'변수명: {parameter[0]}')
    print(f'변숫값: {parameter[1].data}')

summary(net, (2,))

inputs = torch.tensor(x2).float().to(device)

lr = 0.001

net = Net(n_input, n_output)
net.to(device)

criterion = nn.MSELoss()

optimizer = optim.SGD(net.parameters(), lr=lr)

num_epochs = 2000

history = np.zeros((0,2))

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels1) / 2.0
    loss.backward()
    optimizer.step()

    if (epoch % 200 == 0):
        history = np.vstack((history, np.array([epoch, loss.item()])))
        print(f'Epoch {epoch} loss: {loss.item():.5f}')

plt.plot(history[1:,0], history[1:,1], 'b')
plt.xlabel('반복 횟수')
plt.ylabel('손실')
plt.title('학습 곡선(손실)')
plt.show()