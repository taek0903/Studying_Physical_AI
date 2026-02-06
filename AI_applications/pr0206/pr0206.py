import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

torch.manual_seed(0)

ddir = r'D:\rokey\AI_applications\pr0206\hymenoptera_data'

data_transformer = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.490, 0.449, 0.411], [0.231, 0.221, 0.230])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.490, 0.449, 0.411], [0.231, 0.221, 0.230])
    ])
}
'''
data_transformer를 딕셔너리 방식으로 하는 이유
학습과 평가의 명확한 역할 분리
코드 중복 방지 및 간결한 루프처리(가장 큰 장점)
공통 전처리 관리의 용이성
'''

img_data = {k: datasets.ImageFolder(os.path.join(ddir, k), data_transformer[k]) for k in {'train', 'val'}}

# ImageFolder 데이터 셋 활용, DataLoader 객체 생성
# batch_size=8, 데이터 섞기 shuffle 활성화, 작업자 수(num_workers = 2) 설정

dloaders = {'train': torch.utils.data.DataLoader(img_data['train'], batch_size=8, shuffle=True, num_workers=0),
           'val': torch.utils.data.DataLoader(img_data['val'], batch_size=8, shuffle=False, num_workers=0)}

dset_sizes = {x: len(img_data[x]) for x in {'train', 'val'}}

classes = img_data['train'].classes

dvc = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def imageshow(img, text=None):
    img = img.numpy().transpose((1,2,0))
    # torch 배열을 numpy 배열로 반환
    # tensor (C,H,W) numpy (H,W,C)

    # 정규화에 사용했던 R,G,B 채널별 평균(mean) 정의    
    avg = np.array([0.490, 0.449, 0.411])
    # 정규화에 사용했던 R,G,B 채널별 표준편차(stddev) 정의 
    stddev = np.array([0.231, 0.221, 0.230])

    # 역정규화 (denormlization) 수행 img = stddev * img + avg
    img = stddev * img + avg

    # 픽셀 값이 [0,1] 범위를 벗어나는 생길 경우를 대비 => 해당 범위 내로 clip함
    img = np.clip(img, 0, 1)

    plt.imshow(img)
    plt.axis('off')

    # 텍스트(제목) 제공되는 경우, 이미지 제목으로 설정하고 싶어
    if text is not None:
        plt.title(text)

# 학습 데이터로더 ('train') 이터레이터, 넥스트 가져옴
d_iter = iter(dloaders['train'])

# 이터레어터에서 다음 미니배치(이미지 텐서와 클래스 레이블) 가져옴
imgs, cls = next(d_iter)

# 미니 배치 이미지들을 하나의 격자 이미지로 만들어 표현
grid = torchvision.utils.make_grid(imgs)

# 격자 이미지와 해당 레이블(cls) 제목 설정 => 화면에 표시
imageshow(grid, text=[classes[x] for x in cls])

# 전이학습(transfer learning) 함수 정의
def finetune_model(pretrained_model, loss_func, optim, epochs=10):
    # 학습시간 기록
    start = time.time()

    # 현재 모델의 가중치(state_dict)를 깊은 복사 => 초기상태 저장
    model_weights = copy.deepcopy(pretrained_model.state_dict())
    
    # 검증 정확도 추적을 위한 변수를 0.0 초기화
    accuracy = 0.0

    # 지정된 epochs 수 만큼 반복하여 학습을 진행함
    for e in range(epochs):
        print(f'epoch_number {e} / {epochs-1}')
        print('='*20)

        for dset in ['train', 'val']:
            if dset == 'train':
                pretrained_model.train()
                # 학습 모드 설정 (Drop out, BatchNorm 활성화)

            else:
                pretrained_model.eval()
                # 평가 모드 설정 (Drop out, BatchNorm 비활성화)
            
            # 에폭 별 손실과 성공횟수 0.0 초기화
            loss = 0.0
            success = 0
        
            # 학습 또는 검증 데이터 로더 순회
        for imgs, tgts in dloaders[dset]:
            # 입력 이미지. 정답 레이블 => 설정된 device로 이동
            imgs = imgs.to(dvc)
            tgts = tgts.to(dvc)

            optim.zero_grad()

            # 학습 모드('train')에서만 gradient 변화도 계산을 활성화 함
            with torch.set_grad_enabled(dset == 'train'):
                ops = pretrained_model(imgs)
                # 순전파 수행, 예측 결과(ops)얻음
                _, preds = torch.max(ops, 1)
                # 예측 결과에서 모델이 예측한 클래스(preds) 찾음(최대값이 있던 위치 indices)
                # _ : 최대값(value): 필요없음
                loss_curr = loss_func(ops, tgts)

                # 학습 모드인경우만 역전파, 가중치 업데이트 수행
                if dset == 'train':
                    loss_curr.backward()
                    optim.step()

            # 배치 손실을 전체 에폭 손실에 누적
            # => 이미지 개수를 곱해서 평균 손실이 총 손실을 누적함
            loss += loss_curr.item() * imgs.size(0)
            # loss_ccurr : 현재 미니배치의 평균 loss 값
            # .item() => 파이선 숫자(float)
            # img.size(0) : batch_size (현재 배치 내 이미지 개)

            # 예측과 정답과 일치하는 개수 세어서 성공 횟수를 누적함
            success += torch.sum(preds == tgts.data)

        # 에폭이 끝난 후, 전체 손실을 데이터 셋 나누어서 평균 에폭 손실 계산
        loss_epoch = loss / dset_sizes[(dset)]
        # dest_sizes[dset] 데이터 셋(dset)의 전체 크기(총 샘플 수)
        # 전체 성공횟수를 데이터 셋 나누어서 에폭 정확도를 계산함
        accuracy_epoch = success.double() / dset_sizes[dset]
        # .double() 텐서의 데이터 타입을 부동소수점

        print(f'{dset} loss in this epoch: {loss_epoch}, accuracy in this epoch: {accuracy_epoch}')
        print(f'Best accuracy: {accuracy}')   

        if dset == 'val' and accuracy_epoch > accuracy:
            accuracy = accuracy_epoch
            model_weights = copy.deepcopy(pretrained_model.state_dict())
        print()

    # 학습 종료시간 계산 >> 총 소요시간 출력
    time_delta = time.time() - start
    print(f'Training fished in {time_delta // 60}mins {time_delta % 60}secs')
    print(f'Best accuracy: {accuracy}')

    # 최고 성능 보였던 시점의 모델 가중치(model_weights) 를 모델에 로드함
    pretrained_model.load_state_dict(model_weights)

    return pretrained_model 

# 모델의 예측결과 시각화 하는 함수 정의
# pretrained_model : 사전학습된 모델, max_num_imgs : 표시할 최대 이미지 수 입력받음

def visualize_predictions(pretrained_model, max_num_imgs=4):
    torch.manual_seed(1) # 난수 생성기 seed 설정

    # 모델의 원래 학습 모드 상태 (True/False) 저장
    was_model_training = pretrained_model.training
    # 모델이 현재 train() 상태인지 eval() 상태인지 기록
    # => 함수 종료된 뒤에 원래 상태로 복구하기 위함
    plt.figure(figsize=(10, 10))
    # 모델을 평가 모드로 설정
    pretrained_model.eval()

    # 시각화할 이미지 카운터 0으로 초기화
    imgs_counter = 0

    # gradient 계산 비 활성화
    with torch.no_grad():
        # 검증 데이터 로더('val') 순회
        for i, (imgs, tgts) in enumerate(dloaders['val']):
            # 입력 이미지. 정답 레이블 => 설정된 device로 이동
            imgs = imgs.to(dvc)
            tgts = tgts.to(dvc)

            ops = pretrained_model(imgs)
            _, preds = torch.max(ops, 1)
        
        # 현재 배치 내에서 모든 이미지에 대해 순회
            for j in range(imgs.size()[0]):
                imgs_counter += 1
                ax = plt.subplot(max_num_imgs//2, 2, imgs_counter)
                # default max_num_imgs=4 => (2,2)
                ax.axis('off')
                ax.set_title(f'pred: {classes[preds[j]]} || target: {classes[tgts[j]]}')
                imageshow(imgs.cpu().data[j])
            
            # 설정된 최대 이미지 수에 도달한다면
                if imgs_counter == max_num_imgs:
                    pretrained_model.train(mode=was_model_training)
                    # 모델의 모드를 원래 상태로 되돌려라
                    plt.tight_layout()
                    plt.show()
                    return
                    # 함수 실행 종료
        pretrained_model.train(mode=was_model_training)
        #  loop를 끝까지 실행했다면 모델의 모드를 원래 상태로 돌려놓아요       

# torchvision.models 에서 alexnet 모델 불러옴
# pretrained=True 설정 ImageNet 데이터셋으로 미리 학습된 가중치 로드함

model_finetuned = models.alexnet(pretrained=True) # deprecated soon (chagned to 'weights')

# 로드된 alexnet 모델의 특징추출기(Convolution layer) 부분인 feature 모듈의 구조 출력
print(model_finetuned.features)

model_finetuned = models.alexnet(pretrained=True)
# 기존 1,000개 클래스 >> 현재 데이터 셋 클래스 개수(2개, 벌/개미) 변경
# 기존 분류기 마지막 레이어 classifier[6] (인덱스 6) 수정
model_finetuned.classifier[6] = nn.Linear(4096, 2)

# 손실함수 cross entropy 정의
loss_func = nn.CrossEntropyLoss()

# optimizer 정의
optim_finetune = optim.SGD(model_finetuned.parameters(), lr=0.0001)

model_finetuned = model_finetuned.to(dvc)

model_finetune = finetune_model(model_finetuned, loss_func, optim_finetune)
model_finetune

visualize_predictions(model_finetune)

'''
전이학습
대규모 데이터셋으로 미리 학습된 모델의 가중치(weight)를 가져와 
부족한 나의 데이터셋에 맞게 재보정(Fine-tuning)하여 사용하는 방법
특징(ex선, 면, 질감 등)을 추출하여 그대로 물려받고 학습속도와 성능 최적화에 좋다.
'''