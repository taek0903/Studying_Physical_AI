import torch            # PyTorch: 딥러닝 모델 구축 및 텐서 연산을 위한 핵심 라이브러리
import torch.nn as nn   # Neural Network 모듈: 레이어(Layer)와 모델을 정의하는 기능
import torch.nn.functional as F # 자주 사용되는 함수 모음 (활성화 함수, 손실 함수 등)
import torch.optim as optim # 옵티마이저(Optimizer) 모듈: 모델 파라미터 학습을 위한 기능
from torch.utils.data import DataLoader # 데이터 로딩 및 배치 처리를 위한 유틸리티
from torchvision import datasets, transforms # 이미지 데이터셋과 데이터 전처리(변환) 기능

import matplotlib.pyplot as plt # 데이터 시각화 (그래프, 이미지 출력)
import numpy as np              # 과학 계산 및 다차원 배열 처리 (NumPy)

# --- Captum (모델 설명 가능성, XAI) 라이브러리 임포트 ---
from captum.attr import IntegratedGradients # 통합 그래디언트(Integrated Gradients) 기법 임포트
from captum.attr import Saliency        # 살리언시 맵(Saliency) 기법 임포트
from captum.attr import DeepLift        # DeepLIFT 기법 임포트
from captum.attr import visualization as viz # 어트리뷰션 결과 시각화를 위한 유틸리티 임포트

np.random.seed(123)
torch.manual_seed(123)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.cn1 = nn.Conv2d(1, 16, 3, 1)
        self.cn2 = nn.Conv2d(16, 32, 3, 1)
        self.dp1 = nn.Dropout(0.10)
        self.dp2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(4608, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.cn1(x)
        x = F.relu(x)
        x = self.cn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dp1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dp2(x)
        x = self.fc2(x)
        op = F.log_softmax(x, dim=1)
        return op
    
def train(model, device, train_dataloader, optim, epoch):
    # 모델을 훈련 모드(training mode) 설정
    # 드롭아웃(Dropout) 이나 배치 정규화와 같은 레이어가 훈련 시에만 작동하도록 활성화
    model.train()

    for b_i, (X,y) in enumerate(train_dataloader):
        # 입력 데이터와 정답 레이블을 설정된 장치(GPU 또는 CPU)로 이동
        X, y = X.to(device), y.to(device)

        optim.zero_grad()
        pred_prob = model(X)
        loss = F.nll_loss(pred_prob, y)
        '''
        손실 함수로 NLL_Loss(Negative Log-Likelihood Loss, 음의 로그 우도 손실) 계산
        F.nll_loss는 주로 모델의 최종 출력이 F.log_softmax일 때 사용
        nll_loss(Log_softmax(x), target) == CrossEntropy(Softmax(x), target)
        => 로그 소프트 맥스 과정을 분리해서 볼 수 있음
        CrossEntropy = log-softmax + NLLLoss
        '''
        loss.backward()
        optim.step()

        if b_i % 10 == 0:
            print('epoch: {} [{}/{} ({:.0f}%)]\t training loss: {:.6f}'.format(
                epoch, b_i * len(X), len(train_dataloader.dataset), # 현재 처리된 데이터 개수와 전체 데이터 개수
                100. * b_i / len(train_dataloader), loss.item())) # 전체 데이터 대비 현재 진행률과 현재 배치 손실
            
def test(model, device, test_dataloader):
    model.eval()

    loss = 0        # 전체 테스트 데이터셋의 총 손실을 누적할 변수
    success = 0     # 올바르게 분류된 샘플의 총 개수 (정답 수)

    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)

            pred_prob = model(X)    # logit값

            # 배치별 손실을 계산하고 전체 손실에 누적
            # (reduction='sum')으로 배치 내 손실 합산
            loss += F.nll_loss(pred_prob, y, reduction='sum').item() 

            # 예측 확률이 가장 높은 클래스 인덱스(dim=1)를 최종 예측으로 선택
            pred = pred_prob.argmax(dim=1, keepdim=True)

            '''
            예측(pred)과 정답 레이블(y)을 비교하여 맞은 개수를 세어 success에 누적
            pred.eq(y.view_as(pred))는 참/거짓 텐서를 반환, sum().item()으로 참의 개수 카운트
            y.view_as(pred) : y(정답 레이블) 텐서 형태를 pred 텐서 형태와 동일(batch_size,1) 변경
            '''
            success += pred.eq(y.view_as(pred)).sum().item()

    loss /= len(test_dataloader.dataset)

    # 테스트 결과 (평균 손실과 정확도)를 보기 좋게 출력합니다.
    print('\nTest dataset: Overall Loss: {:.4f}, Overall Accuracy: {}/{} ({:.0f}%)\n'.format(
        loss, success, len(test_dataloader.dataset), # 전체 데이터셋 크기와 정답 수
        100. * success / len(test_dataloader.dataset))) # 최종 정확도 (%)
    
# 훈련 데이터셋 로더 정의
train_dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(r'D:\rokey\AI_applications\pr0219', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1302,),(0.3069,))
                   ])),
    batch_size=32,
    shuffle=True)

# 테스트 데이터셋 로더 정의
test_dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(r'D:\rokey\AI_applications\pr0219', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1302,),(0.3069,))
                   ])),
    batch_size=500, # 테스트는 훈련보다 큰 배치 크기를 사용하는 경우가 많음
    shuffle=False)

torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ConvNet().to(device)
optimizer = optim = optim.Adadelta(model.parameters(), lr=0.5)

'''
SGD 무조건 일정한 보폭으로 내려감(경사가 급해도, 완만해도)
Adadelta: 경사에 따라 보폭 조절 - 급한 경사: 작은 보폭(안전하게) - 완만한 경사: 큰 보폭(빠르게)
'''

for epoch in range(1,2):
    train(model, device, train_dataloader, optimizer, epoch)
    test(model, device, test_dataloader)

test_samples = enumerate(test_dataloader)
b_i, (sample_data, sample_targets) = next(test_samples)
print(b_i, sample_data.shape, sample_targets.shape)
plt.imshow(sample_data[0][0], cmap='gray', interpolation='none')
plt.show()

sample_data = sample_data.to(device)
print(sample_data[0].shape)     # batch 사이즈[channl, H, W]

print(sample_data[0][0].shape)  # 사진 크기[H, W]

print(model(sample_data))  # 예측값
print(model(sample_data).shape)

print(model(sample_data).data.max(1))           # dim=1 두번째 차원(클래스 차원) 기준 최대값 찾기
print(model(sample_data).data.max(1)[1])        # 최대값 인덱스 텐서
print(model(sample_data).data.max(1)[1].shape)  # torch.Size([500]) batch_size
print(model(sample_data).data.max(1)[1][0])     # tensor(1) 첫번째 이미지에 대한 예측 인덱스

print(f"Model prediction is : {model(sample_data).data.max(1)[1][0]}")
print(f"Ground truth is : {sample_targets[0]}")

captum_input = sample_data[0].unsqueeze(0)  # batch 차원 증가
captum_input.requires_grad=True
'''
batch 차원 증가
captum은 배치 형태의 입력 요규
requires_grad = True
=> 기울기 계산 활성화
=> XAI는 기울기 사용 => 각 pixel의 중요도 계산
'''

# 원본 이미지 준비 및 전처리
'''
sample_data[0]: DataLoader에서 얻은 첫 번째 샘플(텐서 형태)
.cpu().detach().numpy(): PyTorch 텐서를 Numpy 배열로 변환(GPU -> CPU 기울기 계산 그래프에서 분리)
역 정규화(num/2)+0.5 데이터 로더에서 적용된 정규화를 역으로 되돌리는 근사적 역변환
np.transpose(..., (1,2,0)): PyTorch 텐서의 채널순서 (C,H,W)를 (H,W,C)로 변환
'''
tmp_image = np.transpose((sample_data[0].cpu().detach().numpy() / 2)+0.5, (1,2,0))

orig_image = np.tile(tmp_image, (1,1,3))
'''
np.tile(..., (1,1,3)): 단일 채널 이미지를 RGB 3채널로 복제하여 만듦
ConvNet 모델이 1채널을 입력받았지만, 시각화는 보통 3채널 RGB형식에서 더 잘 처리됨
Camtum 시각화 함수  rgb 기대
=> MNIST(흑백, 1채널) => 3채널(복제)

np.tile
[픽셀값] => [픽셀값, 픽셀값, 픽셀값]
차원축 (0,1,2) 따라서 반복할 횟수 지정
같은 값 R,G,B 복사
(1,1,3)
0번축 1: 첫번째 차원(h:높이) 1번만 반복(높이 변화 X)
1번축 1: 두번째 차원(w:너비) 1번만 반복(너비 변화 X)
2번축 3: 세번째 차원(c:채널) 3번 반복(채널 변화 O)
변환전 (hwc) (28,28,1) => (28,28,3)
'''

# Sailiency Map
saliency = Saliency(model)
# Saliency 객체 생성: 모델(ConvNet)의 입력 대비 출력 기울기를 계산하는 도구 정의
gradients = saliency.attribute(captum_input, target=sample_targets[0].item())
'''
Saliency 계산(어트리뷰선)
.attribute(): 입력(captum_input)에 대해 타겟 클래스 예측에 대한 기울기 계산
target=sample_targets[0].item(): 모델이 예측한 클래스를 타겟으로 설정, 
                                 모델이 그 클래스를 예측했는지에 대한 중요도 찾음
'''

gradients = gradients.squeeze().cpu().detach().numpy()  # 텐서 => 넘파이

gradients = np.reshape(gradients, (28,28,1))

_ = viz.visualize_image_attr(gradients, orig_image, method='blended_heat_map', sign='absolute_value',
                             show_colorbar=True, title='Overlayed Gradients')

'''
시각화
viz.visualize_image_attr(): Captum의 시각화 함수
gradients: 어트리뷰션 결과(각 픽셀의 중요도)
orig_image: 원본 이미지(배경으로 사용)
method='vlended_heat_map': 어트리뷰션 맵(히트맵)과 원본 이미지를 겹쳐서 시각화
sign='absolute_value': 양의 기여도와 음의 기여도 구분 없이 기울기의 '절대값'을 중요도로 사용하여 시각화
show_colorbar=True: 중요도 스케일을 표시하는 컬러 바를 출력
'''

plt.imshow(np.tile(gradients/np.max(gradients), (1,1,3)))

# Intergrated Gradients
'''
1. 단계 1: 경로 정의
   - 경로 a = 0 (검은 이미지) 아무런 정보가 없는 시작점(흑백)
   - 경로 a = 1 (원본 이미지)
   - 경로 a = 0.5 둘의 중간 (반 투명 이미지) (컬러: 평균)

2. 단계 2: 경로를 따라서 기울기 계산
   - 각 a 값에서 모델의 기울기 d(F) / d(x_i) 계산

3. 단계 3: 적분(모든 기울기를 합친다)
   - 수치적으로 리만 합으로 근사
   - 현실적으로 적분 안됨 (m개 점을 샘플링, m=50~300)

4. 단계 4: 입력 차이 곱하기
   - (x_i - x'_i) : 해당 픽셀이 기준점에서 얼마나 변했나

5. 리만 합: 직사각형의 넓이의 합 => 적분에 근사하게 기본 도구
'''

integ_grads = IntegratedGradients(model)
# IntegratedGradients 객체 생성: 모델(ConvNet)에 대한 통합 그래디언트 계산 도구 정의

# 통합 그래디언트(IG)계산
attributed_ig, delta = integ_grads.attribute(
    captum_input,                       # 분석할 입력 데이터
    target=sample_targets[0].item(),    # 타겟 클래스 인덱스(모델이 예측한 클래스)
    baselines=captum_input * 0,         # 기준선(baseline) 설정: 모든 픽셀 값을 0으로 설정한 이미지(흑백 이미지의 경우 일반적)
    return_convergence_delta=True       # 수렴 델타 값(오차)을 반환하도록 설정 (IG의 품질 확인 용도)
)

# 결과 후 처리 및 시각화 준비
attributed_ig = attributed_ig.squeeze().cpu().detach().numpy()    # 텐서를 넘파이 배열로 변환하고 불필요한 차원 제거
attributed_ig = np.reshape(attributed_ig, (28,28,1))  # (H,W,C)에 맞게 28*28*1 형태로 재구성

'''
시각화
# viz.visualize_image_attr(): Captum의 시각화 함수
# attributed_ig: IG 어트리뷰션 결과 (각 픽셀의 통합된 중요도)
# orig_image: 원본 이미지 (배경으로 사용)
# method="blended_heat_map": 어트리뷰션 맵과 원본 이미지를 겹쳐서 시각화
# sign="all": 양의 기여도(긍정적 영향)와 음의 기여도(부정적 영향)를 모두 다른 색상으로 표시
# show_colorbar=True: 중요도 스케일을 표시하는 컬러 바를 출력
'''

_ = viz.visualize_image_attr(attributed_ig, orig_image, method="blended_heat_map",sign="all", show_colorbar=True,
                             title="Overlayed Integrated Gradients")

'''
DeepLift
DeepLift(model)
- 출력(target) 의 변화량을 입력의 변화량으로 분해 (체인룰)
- 기여도 : 활성화 변화량, 입력 변화량으로 결정

- 회사 매출 100억 >> 200억 증가
  - 영업       50억 기여 (50%)
  - 마케팅팀   30억 기여 (30%)
  - 개발팀     20억 기여 (20%)

- 이미지 관점에 보면
  - pixel a : 출력 증가하는 데 0.3 기여
  - pixel b : 출력 증가하는 데 0.2 기여
  - pixel c : 출력 증가하는 데 -0.1 기여 (부정적 영향)

'''

deep_lift = DeepLift(model) # DeepLIFT 객체 생성

# DeepLIFT 계산
attributed_dl = deep_lift.attribute(
    captum_input,                       # 분석할 입력 데이터
    target=sample_targets[0].item(),    # 타겟 클래스 인덱스
    baselines=captum_input * 0,         # 기준선 설정: 모든 픽셀 값을 0으로 설정한 이미지
    return_convergence_delta=False      # 수렴 델타 값(오차) 반환을 비활성화
)

# 결과 후처리 및 시각화 준비
attributed_dl = attributed_dl.squeeze(0).cpu().detach().numpy() # 텐서를 넘파이 배열로 변환
# squeeze(0) : 첫번째 배치차원만 제거
attributed_dl = np.reshape(attributed_dl, (28,28,1))    # (H,W,C) 형태로 재구성

'''
시각화
# viz.visualize_image_attr(): Captum의 시각화 함수
# attributed_dl: DeepLIFT 어트리뷰션 결과 (각 픽셀의 기여도)
# orig_image: 원본 이미지 (배경)
# method="blended_heat_map": 어트리뷰션 맵과 원본 이미지를 겹쳐서 시각화
# sign="all": 양의 기여도(타겟 예측에 긍정적)와 음의 기여도(부정적)를 모두 표시
# show_colorbar=True: 중요도 스케일을 표시하는 컬러 바 출력
'''
_ = viz.visualize_image_attr(attributed_dl, orig_image, method="blended_heat_map",sign="all",show_colorbar=True,
                             title="Overlayed DeepLift")

'''
요약 정리

- 통합그래디언트(IG)는 0(가상의 기준선 baseline) 에서 원본이미지(input 실제 입력) 까지의 가상의 경로(path) 따라서 모델의 예측 변화 분석, 수치화

- 1) 0에서 시작(기준선) baseline 아무 정보도 없는 상태
- 2) 조금씩 밝히기 (경로 이동)    
  - 여기서 경로이동이란 예측값 변화(target) 에 대한 기울기(gradient) 계산
- 3) 영향력 누적(적분 >> 수치화)
  - 경로를 따라 계산된 모든 기울기 값 누적(적분) (리만 합)
  - 즉, 각 픽셀이 기준선(0) 에서 실제 이미지로 변하는 데 있어 총체적으로 얼마나 기여했나 수치화
'''

'''
코드 오류로 애먹은 곳
sample_data에 device 설정을 해줘야 했음
=> device 설정을 안했을 때 input type가 cpu에 있고 모델은 gpu에 있기 때문에
같은 곳에서 계산할 수 없음 그렇기 대문에 sample_data를 gpu로 옮겨주어서 계산을 할 수 있게 만들어 준다.
'''