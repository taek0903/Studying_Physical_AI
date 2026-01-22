import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchinfo import summary
from torchviz import make_dot
from sklearn.datasets import load_digits  # 손글씨 숫자 데이터 (0~9)
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from torch.utils.data import DataLoader
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

# # 활성화 함수와 ReLU함수

# # ReLu 함수의 그래프
# relu = nn.ReLU()    # ReLU 함수 인스턴스 생성
# x_np = np.arange(-2.0, 2.1, 0.25)
# x = torch.tensor(x_np).float()
# y = relu(x)

# plt.plot(x.data, y.data)
# plt.title('ReLU 함수')
# plt.grid(True)
# plt.show()

# x_np = np.arange(-2.0, 2.1, 0.25)
# y_np = np.arange(-1.0, 3.1, 0.25)
# x = torch.tensor(x_np).float()
# y = torch.tensor(y_np).float()

# 데이터 준비 1(Dataset을 활용해 불러오기)
import torchvision.datasets as datasets

data_root = './data'

train_set0 = datasets.MNIST(
    # 원본 데이터를 다운로드받을 디렉터리 지정
    root = data_root,
    # 훈련 데이터인지 또는 검증 데이터인지
    train=True,
    # 원본 데이터가 없는 경우, 다운로드를 실행하는지 여부
    download=True
)

# 데이터 건수 확인
print(f'데이터 건수: {len(train_set0)}')

# # 첫번째 요소 가져오기
# image, label = train_set0[0]

# # 입력 데이터를 이미지로 출력
# plt.figure(figsize=(1,1))
# plt.title(f'{label}')
# plt.imshow(image, cmap='gray_r')
# plt.axis('off')
# plt.show()

plt.figure(figsize=(10,3))
for i in range(20):
    ax = plt.subplot(2, 10, i + 1)

    # image와 label을 취득
    image, label = train_set0[i]

    plt.imshow(image, cmap='gray_r')
    ax.set_title(f'{label}')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# 데이터 준비 2 (Transforms를 활용한 데이터 전처리)
# step1. ToTensor 사용하기  
transform1 = transforms.Compose([
    # 데이터를 텐서로 변환
    transforms.ToTensor()
])
# transforms.Compose : 데이터를 변환하는 과정을 동시에 진행한다.

train_set1 = datasets.MNIST(
    root=data_root, train=True, download=True,
    transform=transform1)

image, label = train_set1[0]
print(f'입력 데이터 타입: {type(image)}')
print(f'입력 데이터 shape: {image.shape}')
print(f'최소값: {image.data.min()}')
print(f'최대값: {image.data.max()}')

# 스텝 2.Normalize 사용하기
transform2 = transforms.Compose([
    # 데이터를 텐서로 변환
    transforms.ToTensor(),

    # 데이터 정규화
    transforms.Normalize(0.5,0.5)
])

train_set2 = datasets.MNIST(
    root = data_root, train=True, download=True,
    transform=transform2
)

# 변환 결과 확인

image, label = train_set2[0]
print('shape : ', image.shape)
print('최솟값 : ', image.data.min())
print('최댓값 : ', image.data.max())

# 스텝 3. Lambda 클래스를 사용해 1차원으로 텐서 변환하기
transform3 = transforms.Compose([
    # 데이터를 텐서로 변환
    transforms.ToTensor(),

    # 데이터 정규화
    transforms.Normalize(0.5, 0.5),

    # 현재 텐서를 1계 텐서로 변환
    transforms.Lambda(lambda x: x.view(-1))
]) 

train_set3 = datasets.MNIST(
    root = data_root, train=True,
    download=True, transform=transform3
)
image, label = train_set3[0]
print('shape : ', image.shape)
print('최솟값 : ', image.data.min())
print('최댓값 : ', image.data.max())

# 데이터 준비 2
transform = transforms.Compose([
    # (1) 데이터를 텐서로 변환
    transforms.ToTensor(),
    # (2) 데이터 정규화
    transforms.Normalize(0.5, 0.5)
    # (3) 1계 텐서로 변환
])

# 훈련용 데이터셋 정의
train_set = datasets.MNIST(
    root=data_root, train=True,
    download=True, transform=transform
)

# 검증용 데이터셋 정의
test_set = datasets.MNIST(
    root=data_root, train=False,
    download=True, transform=transform
)

# 데이터 준비3 (데이터로더를 활용한 미니배치 데이터 생성)

batch_size = 500

# 훈련용 데이터로더 셔플 적용
train_loader = DataLoader(
    train_set, batch_size=batch_size,
    shuffle=True
)

# 검증용 데이터로더 셔플 비적용
test_loader = DataLoader(
    test_set, batch_size=batch_size,
    shuffle=False
)

# 몇개의 그룹으로 데이터를 가져올 수 있는가
print(f'총 미니 배치 개수: {len(train_loader)}')

# 데이터로더로부터 가장 처음 한 세트를 가져옴
for images, labels in train_loader:
    break

print('images의 shape:', images.shape)
print('labels의 shape:', labels.shape)

plt.figure(figsize=(10,3))
for i in range(20):
    ax = plt.subplot(2, 10, i+1)

    # 넘파이로 배열로 변환
    image = images[i].numpy()
    label = labels[i]

    # 이미지의 범위를 [0,1]로 되돌림
    image2 = (image+1)/2

    # 이미지 출력
    plt.imshow(image2.reshape(28, 28),cmap='gray_r')
    ax.set_title(f'{label}')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# 입력 차원수
n_input = 28*28

# 출력 차원수
# 분류 클래스 수는 10
n_output = len(set(list(labels.data.numpy())))

# 은닉층의 노드 수
n_hidden = 128

# 결과 확인
print(f'n_input: {n_input}  n_hidden: {n_hidden} n_output: {n_output}')

# 모델 정의
# 784입력 10출력 1은닉층의 신경망 모델

class Net(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super().__init__()

        # 은닉층 정의(은닉층 노드 수: n_hidden)
        self.l1 = nn.Linear(n_input, n_hidden)

        # 출력층 정의
        self.l2 = nn.Linear(n_hidden, n_output)

        # ReLU 함수 정의
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(x.size(0),-1)
        x1=self.l1(x)
        x2=self.relu(x1)
        x3=self.l2(x2)
        return x3

# 난수 고정 
torch.manual_seed(123)
torch.cuda.manual_seed(123)

# 모델 인스턴스 생성
net = Net(n_input, n_output, n_hidden)

# 모델을 GPU로 전송
net = net.to(device)

# 학습률
lr = 0.01

# 최적화 알고리즘: 경사 하강법
optimizer= torch.optim.SGD(net.parameters(), lr=lr)

# 손실 함수: 교차 엔트로피 함수
criterion = nn.CrossEntropyLoss()

# 데이터로더에서 가장 처음 항목을 취득
for images, labels in train_loader:
    break

# 데이터로더에서 취득한 데이터를 GPU로 보냄
inputs = images.to(device)
labels = labels.to(device)

# 예측 계산
outputs = net(inputs)

# 결과 확인
print(outputs)

# 손실 계산
loss = criterion(outputs, labels)

# 손실값 가져오기
print(loss.item())

g = make_dot(loss, params=dict(net.named_parameters()))
display(g)

# 경사 계산 실행
loss.backward()

# 결과
w = net.to('cpu')
print(w.l1.weight.grad.numpy())
print(w.l1.bias.grad.numpy())
print(w.l2.weight.grad.numpy())
print(w.l2.bias.grad.numpy())

# 경사 하강법 적용
optimizer.step()

# 파라미터 값 출력
print(net.l1.weight)
print(net.l1.bias)

# 반복 계산
# 난수 고정
torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True

# 학습률
lr = 0.01

# 모델 초기화
net = Net(n_input, n_output, n_hidden).to(device)

# 손실 함수 : 교차 엔트로피 함수
criterion = nn.CrossEntropyLoss()

# 최적화 함수: 경사 하강법
optimizer = optim.SGD(net.parameters(), lr=lr)

# 반복 횟수
num_epochs = 100

# 평가 결과 기록
history = np.zeros((0,5))

from tqdm import tqdm

for epoch in range(num_epochs):
    train_acc, train_loss = 0, 0
    val_acc, val_loss = 0, 0
    n_train, n_test = 0, 0

    for inputs, labels in tqdm(train_loader):
        n_train +=len(labels)

        # GPU로 전송
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 경사 초기화
        optimizer.zero_grad()

        # 예측 계산
        outputs = net(inputs)

        # 손실 계산
        loss = criterion(outputs, labels)

        # 경사 계산
        loss.backward()

        # 파라미터 수정
        optimizer.step()

        # 예측 라벨 산출
        predicted = torch.max(outputs, 1)[1]

        # 손실과 정확도 계산
        train_loss += loss.item()
        train_acc +=(predicted == labels).sum().item()

    for inputs_test, labels_test in test_loader:
        n_test += len(labels_test)

        inputs_test = inputs_test.to(device)
        labels_test = labels_test.to(device)

        # 예측 계산
        outputs_test = net(inputs_test)

        # 손실 계산
        loss_test = criterion(outputs_test, labels_test)

        # 예측 라벨 산출
        predicted_test = torch.max(outputs_test, 1)[1]

        # 손실과 정확도 계산
        val_loss += loss_test.item()
        val_acc += (predicted_test==labels_test).sum().item()

    # 평가 결과 산출, 기록
    train_acc = train_acc / n_train
    val_acc = val_acc / n_test
    train_loss = train_loss * batch_size / n_train
    val_loss = val_loss * batch_size / n_test
    print (f'Epoch [{epoch+1}/{num_epochs}], loss: {train_loss:.5f} acc: {train_acc:.5f} val_loss: {val_loss:.5f}, val_acc: {val_acc:.5f}')
    item = np.array([epoch+1 , train_loss, train_acc, val_loss, val_acc])
    history = np.vstack((history, item))

# 손실과 정확도 확인

print(f'초기상태 : 손실 : {history[0,3]:.5f}  정확도 : {history[0,4]:.5f}' )
print(f'최종상태 : 손실 : {history[-1,3]:.5f}  정확도 : {history[-1,4]:.5f}' )

# 학습 곡선 출력(손실)

plt.plot(history[:,0], history[:,1], 'b', label='훈련')
plt.plot(history[:,0], history[:,3], 'k', label='검증')
plt.xlabel('반복 횟수')
plt.ylabel('손실')
plt.title('학습 곡선(손실)')
plt.legend()
plt.show()

# 학습 곡선 출력(정확도)

plt.plot(history[:,0], history[:,2], 'b', label='훈련')
plt.plot(history[:,0], history[:,4], 'k', label='검증')
plt.xlabel('반복 횟수')
plt.ylabel('정확도')
plt.title('학습 곡선(정확도)')
plt.legend()
plt.show()

# 데이터로더에서 처음 한 세트 가져오기
for images, labels in test_loader:
    break

# 예측 결과 가져오기
inputs = images.to(device)
labels = labels.to(device)
outputs = net(inputs)
predicted = torch.max(outputs, 1)[1]

# 처음 50건의 이미지에 대해 "정답:예측"으로 출력

plt.figure(figsize=(10, 8))
for i in range(50):
  ax = plt.subplot(5, 10, i + 1)

  # 넘파이 배열로 변환
  image = images[i]
  label = labels[i]
  pred = predicted[i]
  if (pred == label):
    c = 'k'
  else:
    c = 'b'

  # 이미지의 범위를 [0, 1] 로 되돌림
  image2 = (image + 1)/ 2

  # 이미지 출력
  plt.imshow(image2.reshape(28, 28),cmap='gray_r')
  ax.set_title(f'{label}:{pred}', c=c)
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()

# #  은닉층 추가
# class Net2(nn.Module):
#     def __init__(self, n_input, n_output, n_hidden):
#         super().__init__()
#         self.l1 = nn.Linear(n_input, n_hidden)
#         self.l2 = nn.Linear(n_hidden, n_hidden)
#         self.l3 = nn.Linear(n_hidden, n_output)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x1=self.l1(x)
#         x2=self.relu(x1)
#         x3=self.l2(x2)
#         x4=self.relu(x3)
#         x5=self.l3(x4)
#         return x5
    
# torch.manual_seed(123)
# torch.cuda.manual_seed(123)
# net = Net2(n_input, n_output, n_hidden).to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=lr)
# num_epochs = 100
# history2 = np.zeros((0,5))

# for epochs in range(num_epochs):
#     train_acc = 0
#     train_loss = 0
#     val_acc = 0
#     val_loss = 0
#     n_train = 0
#     n_test = 0

#     for inputs, labels in tqdm(train_loader):
#         n_train += len(labels)

#         inputs = inputs.to(device)
#         labels = labels.to(device)

#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         predicted = torch.max(outputs, 1)[1]
#         train_loss += loss.item()
#         train_acc += (predicted == labels).sum().item()

#     for inputs_test, labels_test in tqdm(test_loader):
#         n_test += len(labels)
#         inputs_test = inputs_test.to(device)
#         labels_test = labels_test.to(device)
#         loss_test = criterion(outputs_test, labels_test)
#         predicted_test = torch.max(outputs_test, 1)[1]
#         val_loss += loss_test.item()
#         val_acc += (outputs_test, labels_test).sum().item()

#     train_acc = train_acc / n_train
#     val_acc = val_acc / n_test
#     train_loss = train_loss * batch_size / n_train
#     val_loss = val_loss * batch_size / n_test
#     print (f'Epoch [{epoch+1}/{num_epochs}], loss: {train_loss:.5f} acc: {train_acc:.5f} val_loss: {val_loss:.5f}, val_acc: {val_acc:.5f}')
#     item = np.array([epoch+1 , train_loss, train_acc, val_loss, val_acc])
#     history2 = np.vstack((history2, item))

# print(f'초기상태 : 손실 : {history2[0,3]:.5f}  정확도 : {history2[0,4]:.5f}' )
# print(f'최종상태 : 손실 : {history2[-1,3]:.5f}  정확도 : {history2[-1,4]:.5f}' )

# plt.plot(history2[:,0], history2[:,1], 'b', label='훈련')
# plt.plot(history2[:,0], history2[:,3], 'k', label='검증')
# plt.xlabel('반복 횟수')
# plt.ylabel('손실')
# plt.title('학습 곡선(손실)')
# plt.legend()
# plt.show()

# plt.plot(history2[:,0], history2[:,2], 'b', label='훈련')
# plt.plot(history2[:,0], history2[:,4], 'k', label='검증')
# plt.xlabel('반복 횟수')
# plt.ylabel('정확도')
# plt.title('학습 곡선(정확도)')
# plt.legend()
# plt.show()

# # 칼럼 경사 소실과 ReLU
# class Net3(nn.Module):
#     def __init__(self, n_input, n_output, n_hidden):
#         super().__init__()
#         self.l1 = nn.Linear(n_input, n_hidden)
#         self.l2 = nn.Linear(n_hidden, n_hidden)
#         self.l3 = nn.Linear(n_hidden, n_output)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x1=self.l1(x)
#         x2=self.sigmoid(x1)
#         x3=self.l2(x2)
#         x4=self.sigmoid(x3)
#         x5=self.l3(x4)
#         return x5
    
# net = Net3(n_input, n_output, n_hidden).to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=lr)

# def fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history):
#     base_epochs = len(history)
#     for epoch in range(base_epochs, num_epochs+base_epochs):
#         train_loss = 0
#         train_acc = 0
#         val_loss = 0
#         val_acc = 0
        
#         # 학습 페이즈
#         count = 0
#         for inputs, labels in tqdm(train_loader):
#             count += len(labels)
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             optimizer.zero_grad()
#             outputs = net(inputs)
#             loss = criterion(outputs, labels)
#             train_loss += loss.item()
#             loss.backward()
#             optimizer.step()
#             predicted = torch.max(outputs, 1)[1]
#             train_acc += (predicted == labels).sum().item()
#             avg_train_loss = train_loss / count
#             avg_train_acc = train_acc / count
        
#         # 예측 페이즈
#         count = 0
#         for inputs, labels in test_loader:
#             count += len(labels)
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             outputs = net(inputs)
#             loss = criterion(outputs, labels)
#             val_loss += loss.item()
#             predicted = torch.max(outputs, 1)[1]
#             val_acc += (predicted == labels).sum().item()
#             avg_val_loss = val_loss / count
#             avg_val_acc = val_acc / count

#         print (f'Epoch [{(epoch+1)}/{num_epochs+base_epochs}], loss: {avg_train_loss:.5f} acc: {avg_train_acc:.5f} val_loss: {avg_val_loss:.5f}, val_acc: {avg_val_acc:.5f}')
#         item = np.array([epoch+1, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc])
#         history = np.vstack((history, item))
#     return history

# # batch size 조정

# # 파이토치 난수 고정
# def torch_seed(seed=123):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)

# # batch_size=500
# # 미니 배치 사이즈 지정
# batch_size_train = 500
# # 훈련용 데이터로더
# # 훈련용이므로 셔플을 적용함
# train_loader = DataLoader(
#     train_set, batch_size = batch_size_train,
#     shuffle = True)
# # 난수 고정
# torch_seed()
# lr = 0.01
# net = Net(n_input, n_output, n_hidden).to(device)
# optimizer = optim.SGD(net.parameters(), lr=lr)
# criterion = nn.CrossEntropyLoss()
# num_epochs = 100
# history6 = np.zeros((0,5))
# history6 = fit(net, optimizer, criterion, 
#                num_epochs, train_loader, test_loader, device, history6)

# # batch_size=200
# batch_size_train = 200
# torch_seed()
# lr = 0.01
# net = Net(n_input, n_output, n_hidden).to(device)
# optimizer = optim.SGD(net.parameters(), lr=lr)
# criterion = nn.CrossEntropyLoss()
# num_epochs = 100
# history3 = np.zeros((0,5))
# history3 = fit(net, optimizer, criterion, 
#                num_epochs, train_loader, test_loader, device, history3)

# # batch_size=100
# batch_size_train = 100
# torch_seed()
# lr = 0.01
# net = Net(n_input, n_output, n_hidden).to(device)
# optimizer = optim.SGD(net.parameters(), lr=lr)
# criterion = nn.CrossEntropyLoss()
# num_epochs = 100
# history4 = np.zeros((0,5))
# history4 = fit(net, optimizer, criterion, 
#                num_epochs, train_loader, test_loader, device, history4)

# # batch_size=50
# batch_size_train = 50
# torch_seed()
# lr = 0.01
# net = Net(n_input, n_output, n_hidden).to(device)
# optimizer = optim.SGD(net.parameters(), lr=lr)
# criterion = nn.CrossEntropyLoss()
# num_epochs = 100
# history5 = np.zeros((0,5))
# history5 = fit(net, optimizer, criterion, 
#                num_epochs, train_loader, test_loader, device, history5)

# # 학습 곡선 비교
# # 학습 곡선 출력(정확도)
# plt.plot(history[:,0], history[:,4], label='batch_size=500', c='k', linestyle='-.')
# plt.plot(history3[:,0], history3[:,4], label='batch_size=200', c='b', linestyle='-.')
# plt.plot(history4[:,0], history4[:,4], label='batch_size=100', c='k')
# plt.plot(history5[:,0], history5[:,4], label='batch_size=50', c='b')
# plt.xlabel('반복 횟수')
# plt.ylabel('정확도')
# plt.title('학습 곡선(정확도)')
# plt.legend()
# plt.show()
'''
주석 처리 부분은 중간에 에러로 터진 상태여서 시간관계상 돌려보지 못함
나중에 수정을 통해서 한번 돌려볼 예정
'''