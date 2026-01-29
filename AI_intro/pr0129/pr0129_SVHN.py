import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchinfo import summary
from torchviz import make_dot
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from tqdm import tqdm
from collections import OrderedDict
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
plt.rcParams["font.family"] = "Malgun Gothic"

torch.manual_seed(42)
np.random.seed(42)

transform_train=transforms.Compose([
    transforms.RandomCrop(32, padding=4),   # padding=4 : 4픽셀 패딩 추가
    transforms.RandomHorizontalFlip(p=0.5),  # 50% 확률로 이미지 좌우 반전
    transforms.ColorJitter(brightness=0.2, 
                           contrast=0.2, saturation=0.2), # 밝기, 대비, 체도 ± 20% 조정
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# 테스트용 전처리 (증강 없음)
transform_test=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

from re import split
from logging import root

train_dataset = datasets.SVHN(
    root='./data',
    split='train',      # 데이터셋 중에서 학습용으로 지정된 부분을 사용
    download=True,
    transform=transform_train
)

test_dataset = datasets.SVHN(
    root='./data',
    split='test',       # 데이터셋 중에서 테스트용으로 지정된 부분을 사용
    download=True,
    transform=transform_test
)

# 데이터 로더 설정
batch_size = 128

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,       # 학습용 데이터는 섞는다. (과적합 방지)
    num_workers=0,
    pin_memory=True     # GPU에 메모리 할당
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,      # 검증용 데이터는 섞지 않는다. => 테스트를 하기 위해서
    num_workers=0,
    pin_memory=True
)

# 샘플 이미지 시각화
# 데이터 셋이 제대로 로드 되었는지 확인하기 위함
class_names=['0','1','2','3','4','5','6','7','8','9']
images, labels = next(iter(train_loader))

def denormalize(tensor):
    return tensor*0.5+0.5
    # 정규화 역변환 : pixel = (normalized * std) + mean

plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i+1)  # 4X4 그리드의 i+1번째 위치

    # 이미지를 Tensor(C, H, W) -> Pixel(H, W, C)
    img = denormalize(images[i]).cpu().numpy().transpose(1, 2, 0)   # GPU 할당된 Tensor를 cpu에 할당

    # 이미지 표시
    plt.imshow(img)
    plt.title(f'Label: {class_names[labels[i]]}')   # 레이블 표시
    plt.axis('off') # 축 숨기기

plt.tight_layout()  # 레이아웃 자동 조정
plt.suptitle('SVHN 학습 데이터 샘플', y=1.02, fontsize=16)
plt.show()

class SVHN_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SVHN_CNN, self).__init__()
        # feature extraction(특징 추출)
        # Conv2d block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32,
                               kernel_size=3, stride=1, padding=1)
        # 파라미터 수 : (3*3*3+1)*32 = 896
        # 출력크기 [batch_size, 32, 32, 32]

        # MaxPool1 : 공간해상도 절반 축소
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 2*2 영역에서 최대값 추출, 2칸씩 이동
        # 출력 크기 [batch_size, 32, 16, 16]
        
        # Conv2d block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, stride=1, padding=1)
        # 파라미터 수 : (3*3*32+1)*64 = 18,496
        # 출력크기 [batch_size, 64, 16, 16]

        # MaxPool2 : 공간해상도 절반 축소
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 출력크기 [batch_size, 64, 8, 8]
        
        # Conv2d block 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=3, stride=1, padding=1)
        # 파라미터 수 : (3*3*64+1)*128 = 73,856
        # 출력크기 [batch_size, 64, 16, 16]

        # MaxPool3 : 공간해상도 절반 축소
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 출력크기 [batch_size, 128, 4, 4]

        # classifier(분류기)
        # fc(fully connected 완전 연결층)
        # 추출된 특징 => 최종분류
        self.fc1 = nn.Linear(in_features=128*4*4, out_features=512)
        # 파라미터 수 (128*4*4+1) * 512 = 1,049,088

        # Dropout : 과적합 방지(학습시 50% 랜덤 비활성화)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)
        # 파라미터 수 (512+1)*10 = 5130
    
    def forward(self, x):
        # feature extraction
        # Conv Block1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Conv Block2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Conv Block3
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # classifier(분류기)
        # flatten(1차원 펼쳐줌)
        x = x.view(x.size(0),-1)
        # x.size(0): batch_size, -1: 나머지 차원 계산해줘

        # fully connected layer 1
        x = self.fc1(x)         # [batch, 2048] => [batch, 512]
        x = F.relu(x)
        x = self.dropout(x)     # drop 적용(학습시에만 적용)

        # fully connected layer 2
        x = self.fc2(x)          # [batch, 512] => [batch, 10]
        return x

model = SVHN_CNN(num_classes=10)
model = model.to(device)

# Hook 사용한 레이어 출력 shape 출력
# hook : 모델의 중간 레이어 출력을 가로채는 매커니즘

# layer 별 출력 저장
layer_outputs = OrderedDict()

def register_hooks(model):
    handles = []

    def hook_fn(module, input, output):
        # module : 현재 실행 중인 레이어 객체, input: layer 입력, output: layer 출력
        layer_name = module.__class__.__name__
        # 클래스 이름을 사용, 레이어 이름 추출
        # 예) Conv2d, ReLU, MaxPool2d

        # 동일한 layer tpye이 여러 개 있을 경우, 번호 추가
        # 중복된 이름 처리
        # 에) Conv2d 이미 2개 있으면 count=2
        count = sum(1 for k in layer_outputs.keys() if layer_name in k)
        if count > 0:
            layer_name = f'{layer_name}_{count+1}'
            # Conv2d_3

        layer_outputs[layer_name] = output.shape
        # 출력 크기를 딕셔너리에 저장

    # 중복이름 처리 1번째 만나는 Conv2d => Conv2d 2번째 만나는 Conv2d => Conv2D_2

    # 모든 하위 모듈(모든 레이어)에 hook 등록
    for name, module in model.named_modules():
        # 전체 모델 자체는 제외 (하위 레어어만 등록)
        if len(list(module.children())) == 0 and module != model:
            # len(list(module.children())) == 0 자식이 없는 레이어
            # Conv2d, MaxPool2d 개별로 등록

            handle = module.register_forward_hook(hook_fn)
            # register_forward_hook : 데이터가 레이어 통과시 hoo_fn 실행해라
            handles.append(handle)

    return handles

# hook 등록
hook_handles = register_hooks(model)
print(len(hook_handles))  #9

# 더미입력 순전파 실행 (hook trigger hook 발생유도)
# 입력 [배치=1, 채널=3, 높이=32, 너비=32]

dummy_input = torch.randn(1,3,32,32).to(device)

with torch.no_grad():
    output = model(dummy_input)

# print(output)
# tensor([[-0.0171, -0.0212,  0.0136,  0.0188, -0.0602,  0.0215,  0.0366,  0.1064,
        #  -0.0219, -0.0834]], device='cuda:0'
for layer_name, shape in layer_outputs.items():
    print(f'{layer_name:<25} {str(tuple(shape)):>30}')
    # layer name output's shape

# hook 제거 (메모리 정리)
for handle in hook_handles:
    handle.remove()

def train_one_epoch(model, criterion, optimizer, dataloader, device):
    model.train()

    running_loss = 0.0      # epoch(학습) 전체 손실 누적
    correct = 0             # 맞춘 샘플 수
    total = 0               # 전체 샘플 수

    pbar = tqdm(dataloader, desc='학습중이에요', leave=False) # Changed from train_loader to dataloader
    # leave=False 학습이 끝나면 진행 바 사라지게 함

    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(device)  # [batch_size, 3, 32, 32]
        labels = labels.to(device)  # [batch_size]

        optimizer.zero_grad()
        outputs = model(inputs)     # [batch, 10]
        loss = criterion(outputs, labels)
        loss.backward() # Fixed typo: bacward() to backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        # loss.item() : 스칼라 값, inputs.size(0) 현재 배치 크기

        _, predicted = outputs.max(1)
        # max1 : dim=1 클래스 차원 최대값 => 인덱스 반환
        # _, : 최대값 무시해
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        # 진행률 표시줄 업데이트
    pbar.set_postfix({
            'loss' : f'{loss.item():.4f}',
            'acc' : f'{100*correct / total:.2f}%',
        })

    # 에폭 평균 손실 및 정확도 계산
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc

# 평가 함수 정의
def evaluate(model, criterion, dataloader, device):
    model.eval()

    running_loss = 0.0      # epoch(학습) 전체 손실 누적
    correct = 0             # 맞춘 샘플 수
    total = 0               # 전체 샘플 수

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='평가중', leave=False) # Changed from train_loader to dataloader

        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
            'loss' : f'{loss.item():.4f}',
            'acc' : f'{100*correct / total:.2f}%',
            })

        avg_loss = running_loss / total
        accuracy = 100 * correct / total
        return avg_loss, accuracy
    
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
        model.parameters(),
        lr = 0.001,
        weight_decay=1e-4   # L2 정규화
)

# 학습 설정 및 실행
num_epochs = 10

# 학습 히스토리 적용
train_losses = []
train_accs = []
test_losses = [] # Changed from 10 to []
test_accs = []

print(num_epochs) # 에폭 수
print(batch_size) # 배치 크기
print(optimizer.param_groups[0]['lr'])  # 학습률
print(optimizer.param_groups[0]['weight_decay'])  # 가중치 감쇠
print(optimizer.__class__.__name__)

best_acc = 0.0

for epoch in range(num_epochs):
    # 학습 단계
    train_loss, train_acc =\
    train_one_epoch(model, criterion, optimizer, train_loader, device)

    # 평가 단계
    test_loss, test_acc =\
    evaluate(model, criterion, test_loader, device)

    # 결과 저장
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    # epoch 결과 출력
    print(f'학습 - loss: {train_loss:.4f}, Accuracy:{train_acc:.2f}%')
    print(f'평가 - loss: {test_loss:.4f}, Accuracy:{test_acc:.2f}%')

    # 최고 성능 모델 저장
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'best_svhn_model.pth')
        # state_dict(): 모델의 모든 파라미터를 딕셔너리로 변환
        print(f' WOW! 당신의 정확도는: {best_acc:.2f}%')

# 학습 곡선 그리기

# 에폭 번호 (x축)
epochs_range = range(1, num_epochs + 1)

# 2x1 서브플롯 생성
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# 왼쪽 그래프: 손실 곡선
ax1.plot(epochs_range, train_losses, 'b-', label='학습 손실', marker='o')
ax1.plot(epochs_range, test_losses, 'r-', label='테스트 손실', marker='s')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('손실 곡선 (Loss Curve)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 오른쪽 그래프: 정확도 곡선
ax2.plot(epochs_range, train_accs, 'b-', label='학습 정확도', marker='o')
ax2.plot(epochs_range, test_accs, 'r-', label='테스트 정확도', marker='s')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('정확도 곡선 (Accuracy Curve)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 최종 성능 요약
print('\n최종 성능 요약:')
print('='*60)
print(f'최종 학습 손실: {train_losses[-1]:.4f}')
print(f'최종 학습 정확도: {train_accs[-1]:.2f}%')
print(f'최종 테스트 손실: {test_losses[-1]:.4f}')
print(f'최종 테스트 정확도: {test_accs[-1]:.2f}%')
print(f'최고 테스트 정확도: {best_acc:.2f}%')

# 예측 결과 시각화
# 테스트 데이터에서 예측 결과 확인

# 최고 성능 모델 로드
model.load_state_dict(torch.load('best_svhn_model.pth'))
# load_state_dict(): 저장된 파라미터를 모델에 로드
model.eval()  # 평가 모드

# 테스트 데이터에서 배치 하나 가져오기
dataiter = iter(test_loader)
images, labels = next(dataiter)

# 모델로 예측
images = images.to(device)
with torch.no_grad():
    outputs = model(images)
    _, predicted = outputs.max(1)

# CPU로 이동 및 역정규화
images = images.cpu()
predicted = predicted.cpu()
labels = labels.cpu()

# 16개 샘플 시각화
plt.figure(figsize=(12, 12))
for i in range(16):
    plt.subplot(4, 4, i+1)

    # 이미지 역정규화 및 차원 변환
    img = denormalize(images[i]).numpy().transpose(1, 2, 0)
    plt.imshow(img)

    # 정답과 예측 표시
    true_label = class_names[labels[i]]
    pred_label = class_names[predicted[i]]

    # 정답이면 파란색, 오답이면 빨간색
    color = 'blue' if true_label == pred_label else 'red'
    plt.title(f'실제: {true_label} / 예측: {pred_label}', color=color, fontsize=10)
    plt.axis('off')

plt.suptitle('테스트 데이터 예측 결과 (파란색: 정답, 빨간색: 오답)',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()

# 정확도 계산
correct = (predicted == labels).sum().item()
total = labels.size(0)
print(f'\n현재 배치 정확도: {100.*correct/total:.2f}% ({correct}/{total})')

import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
# 혼동 행렬 생성 함수
def get_predictions(model, loader, device):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Prediciting'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)

# 혼동 행렬 시각화 함수
def plot_confusion_matrix(y_true, y_pred, class_names):
    # 혼동 행렬 계산
    cm = confusion_matrix(y_true, y_pred)

    # 시각화
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # 클래스별 정확도 계산
    class_accuracy = cm.diagonal() / cm.sum(axis=1) * 100

    print('\n클래스별 정확도:')
    print('=' * 40)
    for i, (name, acc) in enumerate(zip(class_names, class_accuracy)):
        print(f'{name:12s}: {acc:6.2f}%')
    print('=' * 40)

# 예측 및 시각화
y_pred, y_true = get_predictions(model, test_loader, device)
plot_confusion_matrix(y_true, y_pred, class_names)

# 분류 리포트 출력
print('\n\n상세 분류 리포트:')
print('=' * 60)
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

'''
코드리뷰
# layer 별 출력 저장
layer_outputs = OrderedDict()

def register_hooks(model):
    handles = []

    def hook_fn(module, input, output):
        # module : 현재 실행 중인 레이어 객체, input: layer 입력, output: layer 출력
        layer_name = module.__class__.__name__
        # 클래스 이름을 사용, 레이어 이름 추출
        # 예) Conv2d, ReLU, MaxPool2d

        # 동일한 layer tpye이 여러 개 있을 경우, 번호 추가
        # 중복된 이름 처리
        # 에) Conv2d 이미 2개 있으면 count=2
        count = sum(1 for k in layer_outputs.keys() if layer_name in k)
        if count > 0:
            layer_name = f'{layer_name}_{count+1}'
            # Conv2d_3

        layer_outputs[layer_name] = output.shape
        # 출력 크기를 딕셔너리에 저장

    # 중복이름 처리 1번째 만나는 Conv2d => Conv2d 2번째 만나는 Conv2d => Conv2D_2

    # 모든 하위 모듈(모든 레이어)에 hook 등록
    for name, module in model.named_modules():
        # 전체 모델 자체는 제외 (하위 레어어만 등록)
        if len(list(module.children())) == 0 and module != model:
            # len(list(module.children())) == 0 자식이 없는 레이어
            # Conv2d, MaxPool2d 개별로 등록

            handle = module.register_forward_hook(hook_fn)
            # register_forward_hook : 데이터가 레이어 통과시 hoo_fn 실행해라
            handles.append(handle)

    return handles
    
# hook 등록
hook_handles = register_hooks(model)
print(len(hook_handles))  #9

# 더미입력 순전파 실행 (hook trigger hook 발생유도)
# 입력 [배치=1, 채널=3, 높이=32, 너비=32]

dummy_input = torch.randn(1,3,32,32).to(device)

with torch.no_grad():
    output = model(dummy_input)

# print(output)
# tensor([[-0.0171, -0.0212,  0.0136,  0.0188, -0.0602,  0.0215,  0.0366,  0.1064,
        #  -0.0219, -0.0834]], device='cuda:0'
for layer_name, shape in layer_outputs.items():
    print(f'{layer_name:<25} {str(tuple(shape)):>30}')
    # layer name output's shape

# hook 제거 (메모리 정리)
for handle in hook_handles:
    handle.remove()

hook은 모델의 연산 흐름을 바꾸지 않고 특정 레이어의 입력이나 출력을 중간에 가로채서
확인, 저장, 분석할 수 있게 해주는 메커니즘이다.

특히
layer_name = module.__class__.__name__  : 모듈의 class의 이름을 가지고 변수 등록
handle = module.register_forward_hook(hook_fn)

if len(list(module.children())) == 0 and module != model:
# 모듈 내부의 하위 모듈(서브 레이어)가 없고 모듈이  전체 모델 객체 그 자체가 아니여야 한다.
    count = sum(1 for k in layer_outputs.keys() if layer_name in k)
    # layer를 통과할 때 훅으로 가로챈 출력의 키들을 하나씩 k로 꺼내서 확인
    # 키 문자열 k안에 layer의 이름이 부분 문자열로 보함되어 있으면 1을 센다
    # 그렇게 센 개수를 count에 저장한다.
    # 같은 layer_name이 들어간 키가 몇개 있는지 세는 코드
        if count > 0:
            layer_name = f'{layer_name}_{count+1}'
'''