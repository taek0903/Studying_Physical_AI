import torch
import numpy as np

device = torch.device('cpu')

# 2.1 텐서(Tensor) 소개
# 최근의 머신 러닝 시스템은 일반적으로 텐서를 기본 데이터 구조로 사용한다.

# 데이터를 위한 컨테이너라고 생각하면 쉽게 이해할 수 있다.

# NumPy는 훌륭한 프레임워크지만, GPU를 사용하여 수치 연산을 가속화할 수는 없기 때문에 GPU연산이 가능한 PyTorch에서의 Tensor를 사용한다. (디바이스 선택이 수월함.)

# 심층 신경망에서 GPU는 종종 50배 또는 그 이상의 속도 향상을 제공하기 때문에, 안타깝게도 NumPy는 딥러닝 프로그래밍에는 충분치 않다.

# PyTorch 텐서(Tensor)는 개념적으로 NumPy 배열과 동일 : 텐서(Tensor)는 n-차원 배열이며, PyTorch는 이러한 텐서들의 연산을 위한 다양한 기능들을 제공한다.

# 텐서의 특성
# 축의 개수:
# 랭크(Rank)라고도 부르며, 넘파이 배열에서는 ndim을 통해 확인할 수 있다.

# 크기:
# 텐서의 각 축을 따라 얼마나 많은 차원이 있는지를 나타낸 튜플 넘파이 배열에서는 shape을 통해 확인할 수 있다.

# 데이터 타입:
# 텐서에 포함된 데이터 타입이다.
# 타입은 float32, unit8, float64 등이 될 수 있으며 문자열은 지원하지 않는다. dtpye 속성으로 데이터를 확인할 수 있다.

print('---------------------')
# 스칼라(0 Tensor)
# 하나의 숫자만을 담고 있는 텐서 (0차원 텐서)
# 스칼라의 축의 개수 0개

scalar = np.array(10)
print(scalar)           # 10
print(scalar.ndim)      # 0(차원)
print(scalar.shape)     # () 차원이 없으므로 빈 튜플 출력

print('---------------------')
# 백터(Vector) (1 D Tensor)
# 숫자의 배열을 벡터, 1차원 텐서
# 벡터의 축의 개수 : 1개

vector = np.array([1, 2, 3, 4, 5])
print(vector)
print(vector.ndim)  # 1(차원)
print(vector.shape) # (5, ) 1차원 의미

print('---------------------')
# 행렬(Matrix) (2 D Tensor)
# 벡터의 배열을 행렬, 2차원 텐서
# 행렬은 행, 열 2가치 축이 있음

matrix = np.array([[1,2,3],[4,5,6]])
print(matrix)
print(matrix.ndim)  # 2차원 => ndim(차원 확인)
print(matrix.shape) # (2, 3)

print('---------------------')
# 고차원 텐서
# 행렬들을 하나의 새로운 배영을 합치면 숫자로 채워진 직윤면체가 됨 (3 D Temsor)
# 딥러닝에서는 보통 5차원 텐서까지 다룸

tensor_3D = np.arange(24).reshape(2,3,4)
print(tensor_3D)
print(tensor_3D.ndim)   # 차원수 : 3
print(tensor_3D.shape)  # 텐서 형태는 2개의 3*4 행렬 포함

# Tensor의 Shape
# 넘파이의 배열에서 shape를 이용하여 텐서의 크기를 알아볼 경우 return값은 다음과 같다

# 1차원 : (열,)
# 2차원 : (행, 열)
# 3차원 : (깊이, 행, 열)

# 즉, 열의 기준이 되어 차원이 늘어날수록 늘어난 차원의 크기가 열의 앞에 추가된다고 생각하면 된다.
# 머신러닝에서 사용하는 train_data의 shape은 다음과 같다.

# 1. 벡터 데이터: (samples, features) 2D tensor 집값 예측 문제라고 생각하고 주어진 데이터가 100개의 연식, 동네, 역세권의 유무에 따른 데이터라고 하면 (100, 3)크기의 텐서에 저장될 수 있다.

# 2. 이미지: (samples, height, width, channels) 4D tensor 채널 우선방식 과 채널 마지막 방식 으로 나뉘지만 보통의 경우 100 장의 28x28의 컬러 이미지라면 (100, 28, 28, 3)크기의 텐서에 저장될 수 있다.

# 3. 동영상: (samples, Time(frames), height, width, channels) 5D tensor

# FPS (Frame per Second)

# print('---------------------')
# from keras.datasets import mnist

# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# print(train_images.shape)
# print(test_images.shape)
# use_images = train_images[:100,:,:]
# print(use_images.shape)
# 6만장의 데이터 중 위족 절반만 사용하고 싶다면
# train_images[:, :14 , :].shape

# ※ Tensor과 Ndarray
# 텐서 (Tensor)는 NumPy 어레이와 비슷하지만, 텐서는 GPU, TPU와 같은 가속기에서 사용할 수 있고, 텐서는 값을 변경할 수 없습니다.
# 먼저 Pytorch에서 주로 다루는 데이터 자료형인 Tensor과 Numpy에서 주로 다루는 데이터 자료형인 ndarray는 다차원 배열과 배열 생성 시 다양한 데이터 타입 적용이 가능하지만 주요한 차이점이 존재합니다.

# 우선 가장 중요한 차이점은 GPU를 활용한 연산이 가능한가? 이며, 그 외로 중요한 기능들에서 차이점이 있습니다

# Tensor 생성하기
# 텐서는 파이토치에서 스칼라, 벡터, 행렬 같은 개념으로 연산에 사용하는 기본적인 자료구조이며 다차원 배열로 표현할 수 있습니다.
# 파이썬 numpy의 ndarray와 유사합니다.(n차원의 배열 객체)
# 텐서는 n차원으로 구성되어 있는데 0차원은 스칼라, 1차원은 벡터, 2차원은 행렬, 3차원부터 텐서입니다.
# 1차원 텐서, 2차원 텐서, 3차원 텐서라고도 합니다.
# torch 패키지를 임포트하고 tensor 함수를 이용하여 텐서를 생성할 수 있습니다.
print('---------------------')
# 1차원 텐서 생성
a = torch.tensor([1, 2, 3, 4, 5])
print(a)
print(f'텐서크기: {a.size()}')

print('---------------------')
# 2차원 텐서 생성
x1 = torch.tensor([[1,2,3],
                   [4,5,6],
                   [7,8,9]])
print(x1)
print(x1.size())
print(x1.dim())

print('---------------------')
# 3행 2열 리스트로 2차원 텐서 생성
data = [[1,2],[4,5],[6,7]]
print(data)

x2 = torch.tensor(data)
print(x2)
print(x2.shape)

print('---------------------')
# 3차원 텐서 생성

x3 = torch.tensor([[[1,2,3],[4,5,6],[7,8,9]]])
print(x3)
print(f'텐서크기: {x3.size()}')
# 1개의 2차원 행렬(3*3)
# 첫번쨰 차원(깊이) 크기 : 1
# 두번째 차원(행) 크기 : 3
# 세번째 차원(열) 크기 : 3

print('---------------------')
# 넘파이 배열과 파이토치 텐서간 변환
x4 = [[1,2,3],[4,5,6],[7,8,9]]
data_np = np.array(x4)
torch.from_numpy(data_np)

print('---------------------')
# 특정 값으로 텐서 생성

# 0으로 채워진 텐서 생성
zeros_tensor = torch.zeros(2,3)
# 1로 채워진 텐서 텐서 생성
ones_tensor = torch.ones(2,3)
# 특정 값으로 채워진 텐서 생성 
sevens_tensor = torch.full((2,3),7)

print(zeros_tensor)
print(ones_tensor)
print(sevens_tensor)

# random한 텐서 생성
print('---------------------')
# 정규분포를 따르는 무작위한 텐서 생성
random_tensor = torch.randn((2,3))
print(random_tensor)
# torch.randn_like() : 주어진 텐서와 동일한 크기, 데이터타입을 가진 새로운 텐서 생성
# 크기 유지(예제에서는 2*3, 독립적인 무작위 값, 데이터 타입일치(예제에서는 실수), gpu 호환)
tensor_like = torch.randn_like(random_tensor)
print(tensor_like)

print('---------------------')
# rand(행, 열) 0-1 사이의 실수값으로 채워진 텐서
tensor_random = torch.rand(2,3)
print(tensor_random)

print('---------------------')
torch_arange = torch.arange(1,7).reshape(2,3)
print(torch_arange)

# 텐서 속성

# 텐서의 차원을 확인하는 dim() 메서드가 있습니다. 이는 텐서의 rank라고도 불립니다.
# 텐서의 형상(shape)을 확인하는 shape 속성이 있습니다. 이는 각 차원의 크기를 튜플 형태로 반환합니다.
# 텐서의 크기를 확인하는 size() 메서드도 있습니다. 이는 shape와 동일한 정보를 반환합니다.
# 데이터타입 자료형을 확인하는 dtype 속성은 datatype() 속성이며, 이를 통해 텐서의 데이터 타입(예: 정수, 실수, 불리언 등)을 알 수 있습니다.
# 텐서가 저장된 장치(CPU 또는 GPU)를 확인하는 device 속성이 있습니다.
# 텐서의 총 원소 개수를 반환하는 numel() 메서드도 유용합니다.

print('---------------------')
x = torch.randn(3,4,5)
# 3*4*5 크기의 텐서 생성

print(x.dim())
print(x.shape)
print(x.size())
print(x.dtype)
print(x.device)
print(x.numel())    # tensor 총 원소 개수 반환(numel : number of elemnets)

print('---------------------')
# 1차원 텐서
a = torch.tensor([1,2,3,4,5])
print(a.size())
print(a.shape)
print(a.dim())
print(a.ndimension())
print(a.dtype)

print('---------------------')
# 2차원 텐서
x4 =torch.FloatTensor([[1,2],[3,4]])
print(x4.size())
print(x4.shape)
print(x4.dim())
print(x4.ndimension)
print(x4.dtype)

print('---------------------')
# 3차원 텐서
x5 = torch.tensor([[[1,2,3],[4,5,6],[7,8,9]]])
print(x5.size())
print(x5.shape)
print(x5.dim())
print(x5.ndimension())
print(x5.dtype)

# 텐서 생성

# 행렬 곱셉 기본 원리
# 첫번째 행렬의 열수와 두번째 행렬 행수가 일치해야 행렬 곱셈 가능

print('---------------------')
# [2,3]과 [3,2] 텐서 생성
a = torch.tensor([[1,2,3],[4,5,6]])
b = torch.tensor([[7,8],[9,10],[11,12]])
c = torch.matmul(a,b)
print(c, c.shape)

print('---------------------')
tensor1 = torch.tensor([1,2,3])
tensor2 = torch.zeros(3,3)
tensor3 = torch.ones(2,2)
tensor4 = torch.rand(3,3)

# tensor 조작
reshaped = tensor4.reshape(1,9)
transposed = tensor4.t()

print(f'original: {tensor4}')
print(f'reshape: {reshaped}')
print(f'transposed: {transposed}')

print('---------------------')
# 실수형과 정수형 텐서 생성

float_tensor = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32)
print(float_tensor)
int_tensor = torch.tensor([1,2,3], dtype=torch.int64)
print(int_tensor)

# 텐서 연산

# 기본 연산
# 덧셈과 뺄셈: 같은 크기의 텐서 간에 요소별로 수행됩니다.
# 요소별 곱셈: 두 텐서의 대응하는 요소끼리 곱합니다.
# 스칼라 곱: 텐서의 모든 요소에 스칼라 값을 곱합니다.
# 행렬 곱셈
# 행렬 곱셈은 torch.matmul() 또는 @ 연산자를 사용합니다.
# 왼쪽 행렬의 열 수와 오른쪽 행렬의 행 수가 일치해야 합니다.
# 결과 행렬의 크기는 (왼쪽 행렬의 행 수) x (오른쪽 행렬의 열 수)가 됩니다.
# 고급 연산
# 전치(Transpose): 행과 열을 바꾸는 연산으로, t() 메서드를 사용합니다.
# 차원 축소: sum(), mean() 등의 함수로 특정 차원을 따라 값을 집계합니다.
# 브로드캐스팅: 크기가 다른 텐서 간의 연산을 자동으로 조정합니다.
# 비선형 활성화 함수
# ReLU, Sigmoid, Tanh 등의 함수를 텐서에 적용하여 비선형성을 도입합니다.
# 이러한 연산들은 신경망의 순전파와 역전파 과정에서 핵심적인 역할을 합니다.

# 효율적인 텐서 연산은 딥러닝 모델의 성능과 학습 속도에 직접적인 영향을 미칩니다.
print('---------------------')
# 1차 텐서 덧셈
a = torch.tensor([1,2,3])
b = torch.tensor([4,5,6])

c = a + b
print(c)
print(c.shape)

# 1차 텐서 뺄셈
c = a - b
print(c)
print(c.shape)

# 기본 연산
addition = a + b
multiplication = a * b
dot_product = torch.dot(a,b) # torch.dot(a,b) a=[1,2,3] b=[4,5,6] 1*4+2*5+3*6=32
print(addition)
print(multiplication)
print(dot_product)

# print('---------------------')
# # cpu gpu 장치 선택
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tensor = torch.rand(3,3).to(device)
# print(f'Tensor is on {device}')

print('---------------------')
# 텐서 곱셈
# 행렬 곱 matual 주의할 점
# => 행렬 a의 열과 b의 행의 수가 같아야 함

# 텐서 나누기
a = torch.tensor([10,20,30])
b = torch.tensor([2,4,5])

result = a / b
result2 = torch.div(a,b) # div : divide
print(result)
print(result2)

# 텐서 제곱
result3 = a ** 2
result4 = torch.pow(a,2)

print(result3)
print(result4)

print('---------------------')
# 텐서 요소 합
tensor = torch.tensor([[1,2,3],[4,5,6]])
tensor_sum = torch.sum(tensor)

# 텐서 차원 합계
row_sum = torch.sum(tensor, dim=0)  # 행 기준, 열의 합
col_sum = torch.sum(tensor, dim=1)  # 열 기준, 행의 합
print(row_sum)  # [1+4, 2+5, 3+6]
print(col_sum)  # [1+2+3, 4+5+6]

# 텐서 최대값
tensor_max=torch.max(tensor)
print(tensor_max)

# 텐서 최소값
tensor_min=torch.min(tensor)
print(tensor_min)

# 3보다 큰 값들만 True(마스킹)
result = tensor > 3
print(result)

# 텐서 연산을 위한 텐서 변환

# 차원 변경
# view, unsqueez(), squeeze()
# 매우 중요한 것은 차원 크기가 무조건 1로 늘어나거나 1인 것만 줄일 수 있다.
print('---------------------')

exam = torch.rand(3,1,4,1)
print(exam.shape)
print(exam.dim())

exam = exam.squeeze(dim=1)  # torch.size()=[3,1,4,1] 차원의 인덱스 0,1,2,3 중 1번 인덱스 제거
print(exam.shape)   # torch.Size([3,4,1])
print(exam.dim())   # 3

exam = exam.squeeze(dim=1)
print(exam.shape)
print(exam.dim())   # 변동 없음 => 앞선 squeeze에서 [3,4,1]이 되었는데 1번 인덱스의 값이 1이 아님

exam = exam.squeeze(dim=-1)
print(exam.shape)   # torch.Size([3,4])
print(exam.dim())   # 2

exam = exam.unsqueeze(dim=0)
print(exam.shape)   # torch.size([1,3,4])
print(exam.dim())   # 3

# 차원 확장하거나 축소 unsqueeze, squeeze
# tensor 확장은 무조건 차원 확장 후, 차원값 확장으로 진행됨
# 확장했으면 확장된 차원 크기 조정하는 매서드 알아보자

print('---------------------')
# torch.repeat(), torch.expand()

# 정수로 난수 생성
original = torch.randint(0,6,(3,2))

temp = original.unsqueeze(dim=0)    # [1,3,2]
# 확장된 차원을 반복해서 늘리고 싶어라 [1,3,2] => [4,3,2]
# size = 1인 차원만 확장가능
# extend = view + broadcasting

temp_expand = temp.expand(4,3,2)
print(temp_expand)
print(temp_expand.shape)
print(temp)
print(temp.size())

# 확장된 차원을 repeat 매서드 늘림
temp_repeat = temp.repeat(4,1,1)    # torch.Size(4,3,2)
# repeat(dim=0, dim=1, dim=2)
# dim=0, 4로 설정 : 첫번쨰 차원 4번 복제
# dim1=1, dim2=1로 설정 :  그대로 유지
temp_repeat2 = temp.repeat(4,2,1)   # torch.Size(4,6,2)
temp_repeat3 = temp.repeat(4,1,2)   # torch.Size(4,3,4)
print(temp_repeat)
print(temp_repeat.shape)
print(temp_repeat2)
print(temp_repeat2.shape)
print(temp_repeat3)
print(temp_repeat3.shape)

# expand는 차원 사이즈가 늘어나도 메모리는 공유되고 있기에
# 하나의 원소값을 변조하면 공유 메모리로 묶여있는 다른 원소도 같이 영향받는다.

# 따라서 expand로 차원을 늘리는 경우는 브로드 캐스팅 방식으로 읽기전용 데이터를 만들 때 최적화 측면에서 사용하는 메서드라 보면 된다.

# 반대로 repeat메서드는 차원사이즈가 늘어나면 복제된 데이터는 완전히 독립적으로 운용되어 원소 변조가 발생하는 쓰기데이터일 때 위 메서드를 사용한다.

# 주요 텐서 변환 함수
# view() 함수
# 텐서의 크기를 변경하는 데 사용됩니다.
# 변경 후에도 텐서 원소의 총 개수는 동일해야 합니다.
# 예: x1_tensor = x_tensor.view(2, 3)는 3x2 텐서를 2x3 텐서로 변환합니다.
# unsqueeze() 함수
# 지정된 위치에 새로운 차원을 추가합니다.
# dim 매개변수로 차원을 추가할 위치를 지정합니다.
# 예: uns_tensor = x_tensor.unsqueeze(0)는 첫 번째 위치에 새 차원을 추가합니다.
# squeeze() 함수
# 크기가 1인 차원을 제거합니다.
# 특정 연산 후 불필요한 차원을 제거할 때 유용합니다.
# 예: s_tensor = uns_tensor.squeeze()는 크기가 1인 모든 차원을 제거합니다.
# 추가적인 텐서 변환 기능
# chunk() 함수
# 텐서를 지정된 수의 덩어리로 나눕니다.
# 예: chunks = torch.chunk(tensor, 3)는 텐서를 3개의 덩어리로 나눕니다.
# split() 함수
# 텐서를 지정된 크기의 덩어리로 나눕니다.
# 예: splits = torch.split(tensor, 2)는 텐서를 2개의 원소를 가진 덩어리들로 나눕니다.
# 이러한 텐서 변환 함수들은 딥러닝 모델 구축 및 데이터 전처리 과정에서 매우 유용하게 사용됩니다.

# 특히 복잡한 신경망 구조에서 텐서의 형태를 조정하거나, 배치 처리를 위해 데이터의 차원을 조정할 때 자주 활용됩니다.

# view() : tensor 크기 변경 시 사용
# 변경한 후에도 텐서 원소 개수가 같아야 함

# randn() 정규분포 텐서 생성
# => 평균 0 표준편차 1
x_tensor = torch.randn(3,2)
tensor = x_tensor.view(2,3)
tensor1 = x_tensor.view(6)
print(tensor)
print(tensor.shape)     # torch.Size([2,3])
print(tensor1)  
print(tensor1.shape)    # torch.Size([6])
tensor_chunk = tensor1.chunk(3) # 3개의 텐서로 나눔
tensor_split = tensor1.split(2) # 각 텐서가 3개의 원소를 가짐
print(tensor_chunk)
print(tensor_split)