# Python Imaging Library (PIL)의 Image 모듈을 불러옴. 이미지 파일을 열고 조작하는 데 사용함.
from PIL import Image
# Matplotlib의 pyplot 모듈을 plt 별칭으로 불러옴. 시각화에 사용함.
import matplotlib.pyplot as plt
# PyTorch 핵심 라이브러리를 불러옴.
import torch
# 신경망 레이어(nn) 모듈을 불러옴.
import torch.nn as nn
# 옵티마이저(optim) 모듈을 불러옴.
import torch.optim as optim
# TorchVision 라이브러리를 불러옴. (이미지 변환 및 데이터셋 등에 사용)
from torchvision import transforms
import torchvision
# 학습에 사용할 장치(Device)를 설정함. CUDA(GPU)가 사용 가능하면 'cuda'를, 아니면 'cpu'를 선택함.
dvc = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_to_tensor(image_filepath, image_dimension=128):
    img = Image.open(image_filepath).convert('RGB')

    # 디버깅 확인용
    plt.figure()
    plt.title(image_filepath)
    plt.imshow(img)

    # 이미지의 최대해상도가 지정된 크기(image_dimenson)보다 작거나 같으면 => 원본사용
    if max(img.size) <= image_dimension:
        img_size = max(img.size)
    # 이미지의 최대해상도가 지정된 크기 보다 크면 => 지정된 크기 사용
    else:
        img_size = image_dimension

    
    # 이미지를 텐서로 변환하기 위해 파이프라인 정의(Compose)
    torch_transformation = transforms.Compose([
        transforms.Resize(img_size),        # 이미지 사이즈 조정
        transforms.ToTensor()               # 픽셀 값 범위 [0,1] 정규화
    ])

    img = torch_transformation(img).unsqueeze(0)    # 배치 차원(차원 0)

    return img.to(dvc, torch.float)

style_image = image_to_tensor(r'D:\rokey\AI_applications\data\vangogh_starry_night.jpg')
content_image = image_to_tensor(r'D:\rokey\AI_applications\data\Tuebingen_Neckarfront.jpg')

# Gran Matrix
def gram_matrix(ip):
    num_batch, num_channels, height, width = ip.size()
    # 입력 텐서의 4가지 차원 (배치, 채널 수, 높이, 너비) 추출

    # 텐서의 공간 차원(높이 * 너비) => 하나의 벡터 (평탄화 flatten)
    # 텐서의 크기(num_batch, num_channel, width * height) => (b,c,c)
    features = ip.view(num_batch, num_channels, height * width)

    # 배치 행렬 곱셈 ()
    gram = torch.bmm(features, features.transpose(1,2))

    # 정규화 (채널 수 * 높이 * 너비)
    # 요소의 총 개수로 나누어 스케일 조정
    return gram.div(num_channels * height * width)

vgg19 = torchvision.models.vgg19(pretrained=True).to(dvc)

vgg19 = vgg19.features
# vgg19의 model의 특징만 가져다 쓰기

# 신경만 활용한 스타일 트랜스퍼
# 가저온 사전 학습 모델의 파라미터 고정(freezing)
for param in vgg19.parameters():
    param.requires_grad_(False)

# pooling작업 maxpool => avgpool
# maxpool 대시 avgpool 사용시 더 부드러운 특징맵 생성
# =>스타일 전송 품질 높아짐

# 합성곱 레이어의 인덱스 저장할 빈 리스트 생성
conv_indices = []

# vgg19 특징 추출 레이더들 (sequnetial module) 순회
for i in range(len(vgg19)):
    if vgg19[i]._get_name() == 'MaxPool2d':
        vgg19[i] = nn.AvgPool2d(kernel_size=vgg19[i].kernel_size,
                                stride=vgg19[i].stride,
                                padding=vgg19[i].padding)
        
    if vgg19[i]._get_name() == 'Conv2d':
        conv_indices.append(i)
        # 해당 레이어의 인덱스(i) => 리스트에 추가함

conv_indices = dict(enumerate(conv_indices,1))
# 수집된 conv_indices 리스트에서 1터 순서를 매긴 dict() 생성

print(conv_indices)

layers = {1: 's', 2: 's', 3: 's', 4: 'sc', 5: 's'}
# s : style loss, c: content loss

# vgg19 모델의 특징 추출기(features) 부분을 nn.ModuleList 변환
# 전체 모델의 완전연결층으로 구성된 것 중 불필요한 뒷부분 잘라내고(clip)
# 핵심적인 특징 추출기만 남기는 '모델 경량화' 작업(실무)

vgg_layers = nn.ModuleList(vgg19)

# 이전에 수집한 conv_indices(합성곱 인덱스 딕셔너리)에서 가장 큰 인덱스(가장 깊은 레이어) 찾음
last_layer_idx = conv_indices[max(conv_indices.keys())]

# vgg 레이어 리스트를 시작부터 마지막 합성곱 레이어까지 자름(+1을 하면 마지막 레이어 포함)
vgg_layers_trimed = vgg_layers[:last_layer_idx+1]

# 잘라넨 레이어들 nn.Sequntial 모듈로 묶어서 최종 모델 구성
# 이 모델은 이미지에서 스타일과 콘텐츠 특칭을 추출하는데 사용
neural_style_transfer_model = nn.Sequential(*vgg_layers_trimed)
# *vgg_layers_trimed => 리스트 안의 데이터들을 언팩해라

# 최적화 대상 이미지(ip_image)를 콘텐츠 이미지 크기와 동일한 랜덤 노이즈로 초기화
ip_image = torch.randn(content_image.data.size(), device=dvc)

plt.figure()
plt.imshow(ip_image.squeeze(0).cpu().detach().numpy().transpose(1,2,0).clip(0,1))
# 텐서에서 배치차원(0번) 제거, 넘파이 변환 위해 
# cpu로 옮겨주고 때어내주어 (h,w,c)로 변환 픽셀 값은 [0,1] 클립하여 출력
# detach : 이 값은 학습 대상이 아니라 비교를 위한 고정된 정답(target)이니까 계산 추적을 멈춰라 선언

# 총 최적화 에폭 300
num_epochs = 300

# 스타일 손실에 부여할 가중치(weight) 설정함. 콘텐츠 손실보다 훨씬 크게 설정
wt_style = 1e6

# 콘텐츠 손실에 부여할 가중치(weight) 설정
wt_content = 1
'''
스타일 가중치를 크게 주는 이유
스타일 손실값 자체가 수학적으로 너무 작아 숫자를 키워주지 않으면
모델이 이를 무시하고 원본 사진만 그대로 유지하려한다.
그렇기 때문에 스타일을 강제로 밀어붙이기 위해서 가중치를 크게 만든다.
'''

# 각 에폭별(학습할 때마다) 스타일 손실 저장할 리스트 초기화
style_losses = []
# 각 에폭별(학습할 때마다) 콘텐츠 손실 저장할 리스트 초기화
content_losses = []

# 옵티마이저 정의 최적화 대상은 in_image 뿐이며, Adam을 사용, 학습률(lr) 0.1 설정
opt = optim.Adam([ip_image.requires_grad_()], lr=0.1)

for curr_epoch in range(1, num_epochs+1):
    ip_image.data.clamp_(0,1)
    # 생성된 이미지의 pixel 값 [0,1] 범위 내로 강제로 유지
    opt.zero_grad()

    epoch_style_loss = 0
    epoch_content_loss = 0
    # 현재 에폭 스타일 손실과 컨텐츠 손실 누적 변수 0으로 쵸기화

    # 손실 계산할 떄 레이어 인덱스(k) 순회
    for k in layers.keys():
        # layer k가 콘텐츠 손실('c') 계산할 때 포함된 경우
        if 'c' in layers[k]:
            # 콘텐츠 이미지 특징 추출 => 변화도 계산에도 제외(detach)
            target = neural_style_transfer_model[:conv_indices[k]+1](content_image).detach()
            # 현재 이미지 특징 추출
            ip = neural_style_transfer_model[:conv_indices[k]+1](ip_image)
            # 콘텐츠(mse) 계산해서 누적
            epoch_content_loss += torch.nn.functional.mse_loss(ip, target)

        # layer k가 스타일 손실('s') 계산할 때 포함될 경우
        if 's' in layers[k]:
            # 스타일 이미지 특징 추출 => 변화도 계산에서 제외(detach)
            target = gram_matrix(neural_style_transfer_model[:conv_indices[k]+1](style_image)).detach()
            # 현재 이미지 특징 추출
            ip = gram_matrix(neural_style_transfer_model[:conv_indices[k]+1](ip_image))
            # 스타일(mse) 계산해서 누적
            epoch_style_loss += torch.nn.functional.mse_loss(ip, target)

    # 누적된 콘텐츠 손실에 가중치(wt_content) 곱함
    epoch_content_loss *= wt_content

    # 누적된 스타일 손실에 가중치(wt_style) 곱함
    epoch_style_loss *= wt_style

    # 최종 총 손실 계산
    total_loss = epoch_content_loss + epoch_style_loss

    # 역전파 수행, 총손실에 대한 변화(gradient) 계산
    total_loss.backward()

    if curr_epoch % 50 == 0:
        print(f'epoch_number {curr_epoch}')
        print(f'content loss = {epoch_content_loss}, style loss = {epoch_style_loss}')
        plt.figure()
        plt.title(f'epoch number {curr_epoch}')

        # 생성된 이미지를 역변환 (clamp, squeeze, numpy, transpose) 시각화
        plt.imshow(ip_image.data.clamp_(0,1).squeeze(0).cpu().detach().numpy().transpose(1,2,0))
        plt.show()

        content_losses.append(epoch_content_loss.item())
        style_losses.append(epoch_style_loss.item())

    opt.step()

plt.plot(range(50,300+1,50), content_losses, label='content loss')
plt.plot(range(50,300+1,50), style_losses, label='style loss')
plt.legend()
plt.show()