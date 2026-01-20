import torch
import numpy as np
import matplotlib.pyplot as plt
from torchviz import make_dot

plt.rcParams["font.family"] = "Malgun Gothic"

device = torch.device('cpu')

# 3.5 데이터 전처리
# 다섯명의 신장과 체중 데이터를 사용한다.
# 1차 함수를 사용해 신장으로 체중을 예측하는 경우, 
# 최적 직선을 구하는 것이 목적이다.

# 코드 3-1 : 다섯 쌍의 학습 데이터
# 샘플 데이터 선언(신장, 체중)
sampleData1 = np.array([
    [166, 58.7],
    [176.0, 75.7],
    [171.0, 62.1],
    [173.0, 70.4],
    [169.0, 60.1]
])

print(sampleData1)

# 코드 3-2: 학습 데이터를 입력 데이터와 정답 데이터로 분할
# 머신러닝 모델에서 사용하기 위해, 신장을 변수 x로. 체중을 변수 y로 함
x = sampleData1[:,0]
y = sampleData1[:,1]

print(x)
print(y)

# 코드 3-3: 산포도 출력
# 산포도 출력 확인
plt.scatter(x, y, c='k', s=50)  # x, y 데이터를 검은색(k) 점으로 그래프에 표시
plt.xlabel('$x$: 신장 (cm)')    # x축 라벨 설정 
plt.ylabel('$y$: 체중 (kg)')    # y축 라벨 설정
plt.title('신장과 체중의 관계') # 그래프 제목 설정
# plt.show()                      # 그래프를 화면에 표시

# 데이터 변환
# 머신러닝 모델에서 데이터는 0에 가까운 값을 갖는 것이 바람직하다. 
# 따라서, x, y 모두 평균값이 0이 되도록 평행이동시켜서 새로운 좌표계를 X, Y로 한다.
X = x-x.mean()
Y = y-y.mean()

print(X)
print(Y)

# 산포도를 통해 결과 확인
plt.scatter(X, Y, c='k', s=50)
plt.xlabel('$X$')
plt.ylabel('$Y$')
plt.title('데이터 가공 후 신장과 체중의 관계')
# plt.show()

# 3.6 예측 계산
# 코드 3-5: X와 Y를 텐서 변수로 변환
# X와 Y를 텐서 변수로 변환
X = torch.tensor(X).float()
Y = torch.tensor(Y).float()

# 결과 확인
print(X)
print(Y)

# 코드 3-6: W와 B의 정의
# 파라미터 정의
# W와 B는 경사 계산을 위해, requirs_grad=True(자동 계산 미분?)로 설정함
W = torch.tensor(1.0, requires_grad=True).float()   # W, B를 1.0으로 초기화하고, W에 대해 나중에
B = torch.tensor(1.0, requires_grad=True).float()   # 기울기를 계산할 수 있게 설정

print(W)
print(B)

# 코드 3-7: 예측 값 Yp의 계산
# 예측 함수는 1차 함수
def pred(X):
    return W * X + B

# 예측 값 계산
Yp = pred(X)

# 결과 확인
print(Yp)

# 3.7 손실 계산
# 코드 3-9: 손실 함수(mse)의 정의
# 평균 제곱 오차 손실 함수
def mse(Yp, Y):
    loss = ((Yp-Y)**2).mean()
    return loss

# 코드 3-10: 손실 계산
# 손실 계산
loss = mse(Yp, Y)

# 결과 표시
print(loss)

# 3.8 경사 계산
# 코드 3-12: 손실(loss)에 대한 W와 B의 경사 계산 결과
# 경사 계산
loss.backward()

# 경사값 확인
print(W.grad)
print(B.grad)

# 3.9 파라미터 수정
# 코드 3-13: 파라미터 값을 계산한 그대로 갱신하는 경우 (오류 발생)
# 학습률 정의
lr = 0.001 # 주로 [0.1, 0.01, 0.001] 3가지 중 하나로 선택한다.

# 이 코드는 에러를 발생시킵니다. (경사 계산이 필요한 변수('W', 'B')에 대해 'in-place' 연산)
# (자신의 값을 직접 수정하는 연산, 예: '-=')을 수행하려고 했다)
try:
    # 경사를 기반으로 파리미터 수정
    W -= lr * W.grad
    B -= lr * B.grad
except RuntimeError as e:
    print(e)

# 코드 3-15: W, B의 값과 경삿값 확인
# 파라미터와 경삿값 확인
print(W)
print(B)
print(W.grad)
print(B.grad)

# 코드 3-16: 반복 계산의 초기화 처리
# 초기화
# W와 B를 변수로 사용
W = torch.tensor(1.0, requires_grad=True).float()
B = torch.tensor(1.0, requires_grad=True).float()

# 반복 횟수
num_epoch = 500

# 학습률
lr = 0.001

# histoy 기록을 위한 배열 초기화
history = np.zeros((0,2))

# 코드 3-17: 루프 처리 구현
# 루프 처리
for epoch in range(num_epoch):
    # 예측 계산
    Yp = pred(Yp, Y)

    # 손실 계산
    loss = mse(Yp, Y)

    # 경사 계산
    loss.backward()

    # 경사를 기반으로 파라미터 수정
    with torch.no_grad():   # 블록 내에서는 계산 그래프 생성이 일시적으로 중단되어 안전하게 수정 가능
        W -= lr * W.grad
        B -= lr * B.grad

    # 경삿값 초기화
    W.grad.zero_()
    B.grad.zero_()

    # 손실 기록
    if (epoch % 10 == 0):
        item = np.array([epoch, loss.item()])   # loss.item()은 loss 텐서 내부에 있는 값이 하나 일때,
        history = np.vstack((history, item))    # 숫자 1개를 꺼내서 float로 바꾸는 함수
        print(f'epoch = {epoch} loss = {loss:.4f}') # loss.item()을 사용하면 기울기 계산용 그래프랑 분리

# 코드 3-18: 최종 파라미터 값과 손실 값 출력
# 최종 파라미터 값
print(f'W {W.data.numpy()}')
print(f'B {B.data.numpy()}')

# 손실 확인
print(f'초기상태: 손실 : {history[0,1]:.4f}')   # history 배열의 첫 번째 행, 두 번째 열의 값(초기 손실)을 출력
print(f'최종상태: 손실 : {history[-1,1]:4f}')   # history 배열의 마지막 행, 두 번째 열의 값(최종 손실을 출력)

# 코드 3-19: 손실 값을 통한 학습 곡선의 출력
# 학습 곡선 출력(손실)
plt.plot(history[:,0], history[:,1], 'b')   # X축 : history의 모든 행의 첫 번째 열(반복 횟수), Y축: 모든 행의 두 번째 열(손실)
plt.xlabel('반복횟수')
plt.ylabel('손실')
plt.title('학습 곡선(손실)')
# plt.show()

# 산포도에 회귀 직선을 동시에 출력함
# x의 범위를 구함(Xrange)
X_max = X.max()
X_min = X.min()
X_range = np.array((X_min, X_max))
X_range = torch.from_numpy(X_range).float()

# 이와 대응하는 예측값 y를 구함
Y_range = pred(X_range)
print(Y_range.data)

# 코드 3-20: 산포도와 상관 직선 동시 출력
# 그래프 출력
plt.scatter(X, Y, c='k', s=50)
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_range.data, Y_range.data, lw=2, c='c') #lw = linewidth
plt.title('신장과 체중의 상관 직선(가공 후)')
# plt.show()

# 3.12 최적화 함수와 step 함수 이용하기
# 코드 3-22: 최적화 함수를 이용한 초기화 처리
# 초기화
# W와 B를 변수로 사용
W = torch.tensor(1.0, requires_grad=True).float()
B = torch.tensor(1.0, requires_grad=True).float()

num_epoch = 500

lr = 0.001

import torch.optim as optim

optimizer = optim.SGD([W,B], lr=lr)

history = np.zeros((0,2))

for epoch in num_epoch:
    Yp = pred(X)
    loss = mse(Yp, Y)
    loss.backward()
    optimizer.step()        # with torch.no_grad 와 같은 역할
    optimizer.zero_grad()
    if (epoch % 10 == 0):
        item = np.araay([epoch, loss.item()])
        history = np.vstack((history, item))
        print(f'epoch = {epoch} loss = {loss.item():4f}')

print(f'W {W.data.numpy()}')
print(f'B {B.data.numpy()}')

print(f'초기상태: 손실 : {history[0,1]}')
print(f'최종상태: 손실 : {history[-1,1]}')

# 최적화 함수 튜닝
history_default = history

W = torch.tensor(1.0, requires_grad=True).float()
B = torch.tensor(1.0, requires_grad=True).float()

# optimizer에 momentum = 0.9 옵션 추가
optimizer = optim.SGD([W,B], lr=lr, momentum=0.9)
history_momentum = np.zeros((0,2))

for epoch in range(num_epoch):
    Yp = pred(X)
    loss = mse(Yp, Y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (epoch % 10 == 0):
        item = np.array([epoch, loss.item()])
        history_momentum = np.vstack(history_momentum, item)
        print(f'epoch = {epoch} loss = {loss:.4f}')

# 학습 곡선(손실) 비교 출력
plt.plot(history_default[:,0], history_default[:,1], 'c', label='기본값 설정') # 기본값 SGD 결과 (청록색)
plt.plot(history_momentum[:,0], history_momentum[:,1], 'k', label='momentum=0.9') # momentum 적용 결과 (검은색)
plt.xlabel('반복 횟수')
plt.ylabel('손실')
plt.legend() # 범례 표시
plt.title('학습 곡선(손실)')
# plt.show()