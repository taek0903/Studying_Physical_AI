# 운영 체제와 상호 작용하는 함수(예: 파일 경로 처리)를 불러옴.
import os
# NumPy 라이브러리를 불러옴. 배열 및 수치 연산에 사용함.
import numpy as np
# PyTorch의 핵심 라이브러리를 불러옴.
import torch
# 신경망 레이어(nn) 모듈을 불러옴.
import torch.nn as nn
# 함수형 신경망 연산(F) 모듈을 불러옴.
import torch.nn.functional as F
# PyTorch의 데이터 로더 유틸리티를 불러옴.
from torch.utils.data import DataLoader
# 자동 미분(Autograd)을 위한 Variable 클래스를 불러옴. (최신 PyTorch에서는 텐서가 대체함)
from torch.autograd import Variable
# TorchVision의 이미지 변환(transforms) 모듈을 불러옴. 데이터 전처리에 사용함.
import torchvision.transforms as transforms
# 텐서 이미지를 파일로 저장하는 유틸리티 함수를 불러옴.
from torchvision.utils import save_image
# TorchVision의 데이터셋 모듈을 불러옴.
from torchvision import datasets
from tqdm import tqdm

'''
Gan >> DCGAN (CNN 도입해서 성능개선)
SRGAN(Super Resolution) 해상도 개선(오래된 사진 복원, 의료영상)
CycleGAN (스타일 변환) (두개의 생성기 사용) (여름 풍경 >> 겨울 풍경, 모네스타일 >> 사진)
'''

# 총 학습 에폭(epoch) 횟수를 10으로 설정함.
num_eps=1
# 학습 시 사용할 미니 배치(batch) 크기를 32로 설정함.
bsize=32
# 옵티마이저의 학습률(learning rate)을 0.001로 설정함.
lrate=0.001
# 잠재 공간(latent space)의 차원(dimension)을 64로 설정함. (생성 모델의 입력 크기)
lat_dimension=64
# 입력 및 생성될 이미지의 크기(64x64)를 설정함.
image_sz=64
# 이미지의 채널 수(1: 흑백)를 설정함.
chnls=1
# 학습 진행 상황을 로그로 출력할 간격(미니 배치 수)을 200으로 설정함.
logging_intv=200

class GANGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # 선형 레이어의 출력 해상도 계산 ex)64*64 image 1/4 크기 => 16
        self.inp_sz = image_sz // 4

        # 첫 번째 레이어: 잠재공간(lat_dimension)을 (128*inp_sz) 크기 벡터로 변환하는 선형 레이어 정의
        self.lin = nn.Linear(lat_dimension, 128*self.inp_sz*self.inp_sz)

        # 선형 레이어 출력 => 배치정규화
        self.bn1 = nn.BatchNorm2d(128)

        # 이미지 해상도를 2배로 업샘플링
        self.up1 = nn.Upsample(scale_factor=2)

        # 채널 128개 유지하는 3*3 커널 이용, 합성곱 레이어 정의
        self.cn1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)

        # 배치정규화 레이어 정의 (momentum=0.8 사용)
        self.bn2 = nn.BatchNorm2d(128, 0.8)

        # Leaky ReLU 활성화 함수 정의 (음수 기울기 0.2)
        self.rl1 = nn.LeakyReLU(0.2, inplace=True)
        '''
        Leaky ReLU를 사용하는 이유
        ReLU는 음수 값을 받게 되면, 출력도 0이고 기울기도 0이 되는 문제점이 생긴다.
        즉 역전파때 가중치 업데잍트가 전혀 일어나지 않고 뉴련이 죽어버려 다시 회생하지 못하는 상태가 된다.
        이를 Dying ReLU라고 한다.
        입력값이 음수일 때 0이 아니라 아주 작은 기울기를 줘서 역전파가 끊기지 않고 가중치가
        조금씩 업데이트될 수 있게 해준다.
        GAM: Discriminator(판별자): 모델에서 LeakyReLU를 표준으로 사용한다.
        깊은 신경망: 층이 아주 깊을 때 학습이 잘 안되고 죽는 뉴런이 많아보일 때 대안으로 사용
        '''

        # 이미지 해상도를 2배로 업샘플링
        self.up2 = nn.Upsample(scale_factor=2)        

        # 채널을 128 => 64개로 줄이는 3*3 합성곱 레이어 정의
        self.cn2 = nn.Conv2d(128, 64, 3, stride=1, padding=1)

        # 배치정규화 레이어 정의 (momentum=0.8 사용)
        self.bn3 = nn.BatchNorm2d(64, 0.8)

        # Leaky ReLU 활성화 함수 정의 (음수 기울기 0.2)
        self.rl2 = nn.LeakyReLU(0.2, inplace=0.2)

        # 최종 채널 수를 chnls(1개)로 만드는 3*3 합성곱 레이어 정의
        self.cn3 = nn.Conv2d(64, chnls, 3, stride=1, padding=1)

        # 최종 출력 이미지 픽셀값을 [-1,1] 범위로 제한 : Tahn 활성화 함수
        self.act = nn.Tanh()

    def forward(self, x):
        # 선형 레이어 통과
        x = self.lin(x)
        # 텐서를 (배치 크기, 채널, 높이, 너비) 4차원 형태로 재구성
        x=x.view(x.shape[0], 128, self.inp_sz, self.inp_sz)

        # 배치 정규화 1을 통화
        x = self.bn1(x)

        # 업샘플링 1 통과 => 해상도 2배
        x = self.up1(x)

        # 배치 정규화 2을 통과
        x = self.bn2(x)

        # Leaky Relu 통과
        x = self.rl1(x)

        # 업샘플링 2 => 해상도 2배
        x = self.up2(x)

        # 합성곱 2 통과
        x = self.cn2(x)

        # 배치 정규화 3을 통과
        x = self.bn3(x)

        # Leaky ReLU 통과
        x = self.rl2(x)

        # 최종 합성곱 레이어 3 통과
        x = self.cn3(x)

        # Tanh 활성화 함수 통과 => 최종 출력(out)
        out = self.act(x)

        return out

# GAN 판별자 (Discriminator) 클래스 정의
class GANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # 합성곱, leaky relu, dropput => 하나의 판별 모둘 정의하는 헬퍼 함수
        def disc_module(ip_chnls, op_chnls, bnorm=True):
            # 3*3 합성곱, stride=2, padding=1 해상도 절반으로 줄임
            mod = [nn.Conv2d(ip_chnls, op_chnls, 3, 2, 1),
                    # 3: 3*3, 2: stride, 1: padding
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.25)]
            # 배치 정규화(bnorm)가 요청된 경우 모듈에 추가
            if bnorm:
                mod += [nn.BatchNorm2d(op_chnls, 0.8)]
            
            return mod  # 구성된 모듈 리스트를 반환
        
        # 정의된 판별 모듈(disc_module)들을 연결하여 판별자 모델의 특징 추출부(disc_model)구성
        self.disc_model = nn.Sequential(
            # 첫번째 모듈: 입력채널(chnl)d을 16개로 만들어 줌. 배치정규화를 사용하지 않음(False)
            *disc_module(chnls, 16, bnorm=False),
            # 두번째 모듈: 16채널 => 32채널
            *disc_module(16, 32),
            # 세번째 모듈: 32채널 => 64채널
            *disc_module(32, 64),
            # 네번째 모듈: 64채널 => 128채널
            *disc_module(64, 128)
        )

        # 모델을 4번 통과한 후 해상도가 얼마나 줄었는지 계산 (64=>32=>16=>8=>4)
        # 4번의 다운샘플링 후 최종 해상도 계산
        ds_size = image_sz // 2**4

        # 특징맵을 단일 확률 값으로 변환하는 최종 레이어 정의
        self.adverse_lyr = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 1),
            # 입력값 : 128 * ds_size**2
            # 원래 128개 채널, ds_size * ds_size(feature map)
            # => 1차원 스칼라 변환 => 하나의 출력값(벡터값)
            nn.Sigmoid()    # 출력 (0: 가짜 1: 진짜)
        )
    
    def forward(self, x):
        # 이미지를 특징 추출부(disc_model) 통과
        x = self.disc_model(x)
        # 배치 차원 제외(별개로 두고), 텐서를 평탄화 => 선형레이어 입력 형태로 전환
        # flatten 활용 (view 활용 => 원본 손상 안함)
        x = x.view(x.shape[0], -1)
        # (배치크기, 채널, 높이, 너비) 4차원 텐서 => (배치크기. 전체특징벡터길이) 2차원
        # 최종 선형 및 sigmoid lyr 통과 => 결과물 얻음
        out = self.adverse_lyr(x)

        return out
    
gen = GANGenerator()
disc = GANDiscriminator()

# 손실함수 이진 분류
adv_loss_func = torch.nn.BCELoss()

# MINIST 데이터 셋 => 데이터 로더
dloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        '/content/data/mnist',
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize((image_sz, image_sz)),
                # 이미지를 설정한 이미지 사이즈(64*64) 조정
                transforms.ToTensor(),
                # 이미지를 텐서로 변환
                transforms.Normalize([0.5],[0.5])
            ]
        )
    ),
    batch_size=bsize,
    shuffle=True
)

opt_gen = torch.optim.Adam(gen.parameters(), lr=lrate)
opt_disc = torch.optim.Adam(disc.parameters(), lr=lrate)

save_dir = r'D:\rokey\AI_applications\pr0210'
os.makedirs(save_dir, exist_ok=True)

for ep in range(num_eps):
    loop = tqdm(dloader, desc=f'Epoch {ep+1}/{num_eps}')
    for idx, (images, _) in enumerate(loop):
        # 데이터 로더를 돌면서 미니배치(이미지, 레이블 무시) 처리
        # 진짜 이미지에 대한 정답 레이블 (1,0) 텐서 생성 (변화도 추적 비활성화)
        # images.shape[0] : batch_size, [batch_size, 1]
        good_img = torch.ones(images.size(0), 1)

        # 가짜 이미지에 대한 정답 레이블(1,0) 텐서 생성(변화도 추적 비활성화)
        bad_img = torch.zeros(images.size(0),1)

        # 실제 이미지를 pytorch float tensor 타입의 변수로 변환
        actual_images = images.to(dtype=torch.float32)

        # 생성자(Generator) 훈련 단계
        opt_gen.zero_grad()

        # 정규 분포에서 random noise vector 생성 lat_dimension 크기
        noise = torch.randn(images.size(0), lat_dimension)
        # image.shape[0] 배치크기 (미니배치 이미지 개수, lat_dimension: 잠재공간 벡터의 차원(크기))

        # 생성자 모델에 노이즈(z) 넣어서 가짜 이미지 생성
        gen_images = gen(noise)

        # 생성자 손실 계산 : 생성자 이미지를 진짜 (good_img=1.0) 이라고 속일 수 있는지 평가
        generator_loss = adv_loss_func(disc(gen_images), good_img)

        # 생성자 손실 보면서 역전파 수행
        generator_loss.backward()

        # 생성자 파라미터 업데이트
        opt_gen.step()

        # 판별자 훈련
        opt_disc.zero_grad()

        # 실제 이미지에 대한 판별자 손실 계산 (진짜를 진짜로 판단하도록)
        actual_image_loss = adv_loss_func(disc(actual_images), good_img) 

        # 가짜 이미지에 대한 판별자 손실 계산(가짜를 가짜로 판단하도록)
        # gen_images.detach()로 생성자로부터 변화도 전파를 차단함
        fake_image_loss = adv_loss_func(disc(gen_images.detach()), bad_img)

        # 판별자 손실을 진짜 손실과 가짜 손실의 평균 계산
        discriminator_loss = (actual_image_loss + fake_image_loss) / 2

        # 판별자 손실을 바탕으로 역전파 수행
        discriminator_loss.backward()

        # 판별자 파라미터 업데이트
        opt_disc.step()

        loop.set_postfix(G_loss=generator_loss.item(), D_loss=discriminator_loss.item())
        # 현재까지 완료된 배치의 총 개수 계산
        batch_completed = ep * len(dloader) + idx
        # 현재 진행중인 에폭(한번 학습) * 총 배치 수 + 현재 에폭 내에서 진행 중인 배치 인덱스

        # 로깅 간격(logging_intv)마다 손실 출력하고 이미지 저장
        if batch_completed % logging_intv == 0:
            print(f'epoch number {ep} | batch number {idx}' 
                    f'| generator loss = {generator_loss.item()}' 
                    f'| discriminator loss= {discriminator_loss.item()}')
            file_path = os.path.join(save_dir, f'{batch_completed}.png')
            save_image(gen_images.data[:25], file_path, nrow=5, normalize=True)

import natsort # 자연 정렬 모듈

image_dir = save_dir

# 1. 디렉터리 내 파일 목록을 가져와 natsort로 자연 정렬 (예: image1, image2, image10 순서로 정렬)
# cf.일반적인 문자열 정렬(알파벳, 숫자 순) 1.png, 10.png, 2.png
sorted_files = natsort.natsorted(os.listdir(image_dir))

# 2. 정렬된 파일명에 디렉터리 경로를 결합하여 최종 경로 리스트 생성
image_list = [os.path.join(image_dir, x) for x in sorted_files]

import cv2
import matplotlib.pyplot as plt

loaded_images = []

for path in image_list:
   img = cv2.imread(path)

   if img is None:
      print(f"경로에 오류가 있어 파일을 찾을 수 없습니다: {path}")
      continue
      # 이미지 로드하지 못했으면 건너뛰기

   img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   loaded_images.append(img_rgb)


for i in range(len(loaded_images)):
    plt.figure()

    plt.imshow(loaded_images[i])
    plt.title(f'Displayed Image {image_list[i].split('/')[-1]}')

plt.show()