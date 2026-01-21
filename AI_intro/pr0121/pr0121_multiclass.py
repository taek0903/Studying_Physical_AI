import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torchviz import make_dot
from sklearn.datasets import load_digits  # 손글씨 숫자 데이터 (0~9)
from sklearn.preprocessing import StandardScaler
import seaborn as sns
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

# 데이터 로드 및 탐색
def load_and_explore_data():
    '''
    데이터를 로드하고 기본 정보를 출력하는 함수

    Returns:
        tuple: (X, y, digits)
    '''
    # 데이터 로드 및 탐색
    digits = load_digits()
    X, y = digits.data, digits.target

    print('=' * 60)
    print('데이터셋 정보')
    print('=' * 60)
    print(f'전체 샘플 수: {X.shape[0]:,}개')
    print(f'특성(Feature) 수: {X.shape[1]}개 (8X8 픽셀)')
    print(f'클래스 수: {len(np.unique(y))}개 (0~9)')

    # 클래스별 분표
    print('/클래스별 샘플 수')
    unique, counts = np.unique(y, return_counts=True)   
    # return_counts => 유니크한 값과 몇번 나왔는지 반환

    # zip함수 여러 시퀀스를 같은 인덱스끼리 묶어주는 함수
    for digit, count in zip(unique, counts):    
        print(f'숫자 {digit}: {count}개')

    print('=' * 60)

    return X, y, digits
X, y, digits = load_and_explore_data()

# 샘플 이미지 시각화
# def visualize_samples(digits, n_samples=10):
#     '''
#     랜덤 샘플 이미지를 시각화 하는 함수
    
#     Args:
#         digits: digits 데이터셋 객체
#         n_samples: 표시할 샘플 수
#     '''
#     fig, axes = plt.subplots(2, 5, figsize=(12,5))
#     axes = axes.ravel() # ravel => 다차원 데이터를 1차원 벡터로 펼쳐준다.

#     # 랜덤 인덱스 선택
#     random_indices = np.random.choice(len(digits.images), n_samples, replace=False)
#     # => digits의 이미지 데이터들 중에서 n개의 샘플만큼 중복없이 뽑아줘
#     # => replace=True, 중복을 허용해서, replace=Flase, 중복을 포함해서 뽑아줘

#     for idx, ax_idx in enumerate(random_indices):
#         axes[idx].imshow(digits.images[ax_idx], cmap='gray')
#         axes[idx].set_title(f'Label: {digits.target[ax_idx]}', fontsize=12, fontweight='bold')
#         axes[idx].axis('off')

#     plt.suptitle('Sample Handwritten Digits', fontsize=14, fontweight='bold', y=1.02)
#     plt.tight_layout()
#     plt.show()

# visualize_samples(digits)

# 데이터 전처리 및 분할
def prepare_data(X, y, test_size=0.2, val_size=0.25, random_state=42):
    '''
    데이터를 전처리하고 Train/Validation/Test로 분할

    Args:
        X: 입력데이터
        y: 레이블 데이터
        test_size: 테스트 데이터 비율
        val_size: 검증 데이터 비율 (훈련 데이터 중)
        random_state: 랜덤 시드

    Returns:
        turple: (X_train, X_val, X_test, y_train, y_val, y_test, scaler)
    '''
    # Train+Val과 Test 분리
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # stratify = y 클래스 분포(비율) 유지하는 옵션

    # Train과 Validation 분리
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )

    # 표준화 (평균=0, 표준편차=1)
    scaler = StandardScaler()           # 각 특성을 (값-평균)/표준편차 변환 표준화
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print("\n" + "=" * 60)
    print("데이터 분할 결과")
    print("=" * 60)
    print(f"훈련 데이터: {X_train.shape[0]}개 ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"검증 데이터: {X_val.shape[0]}개 ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"테스트 데이터: {X_test.shape[0]}개 ({X_test.shape[0]/len(X)*100:.1f}%)")
    print(f"\n표준화 후 훈련 데이터 통계:")
    print(f"  평균: {X_train.mean():.4f}")
    print(f"  표준편차: {X_train.std():.4f}")
    print("=" * 60)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(X, y)

# PyTorch 텐서 변환
def numpy_to_tensor(X_train, X_val, X_test, y_train, y_val, y_test):
    '''
    NumPy 배열을 Pytorch 텐서로 변환
    '''
    # 입력: float32
    train_inputs = torch.FloatTensor(X_train).to(device)
    val_inputs = torch.FloatTensor(X_val).to(device)
    test_inputs = torch.FloatTensor(X_test).to(device)

    # 레이블 : int64    =>이유 CrossEntropyLoss 사용하기 위해서
    train_labels = torch.LongTensor(y_train).to(device)
    val_labels = torch.LongTensor(y_val).to(device)
    test_labels = torch.LongTensor(y_test).to(device)

    print("\n텐서 변환 완료")
    print(f"  훈련 입력 shape: {train_inputs.shape}, dtype: {train_inputs.dtype}")
    print(f"  훈련 레이블 shape: {train_labels.shape}, dtype: {train_labels.dtype}")

    return train_inputs, val_inputs, test_inputs, train_labels, val_labels, test_labels

# 텐서 변환
train_inputs, val_inputs, test_inputs, train_labels, val_labels, test_labels = numpy_to_tensor(
    X_train, X_val, X_test, y_train, y_val, y_test
)

# 개선된 신경망 모델 정의
class AdvancedClassifier(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, num_classes, dropout_rate=0.4):
        '''
         Args:
            input_size: 입력 특성 수
            hidden1_size: 첫 번째 은닉층 크기
            hidden2_size: 두 번째 은닉층 크기
            num_classes: 출력 클래스 수
            dropout_rate: 드롭아웃 비율
        '''
        super().__init__()

        # 첫 번째 블록 : Linear -> BatchNorm -> ReLU -> Dropout
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.bn1 = nn.BatchNorm1d(hidden1_size) 
        # BatchNorm1d
        # 미니배치 단위로 각 특성을 정규화하고 학습 가능한 스케일과 
        # 시프트를 적용하여 학습을 안정화하고 속도를 향상
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate) 
        # drop out
        # 학습 중 일부 뉴런을 확률적으로 비활성화하여 과적합을 방지하고 
        # 일반화 성능을 높이는 기법

        # 두 번째 블록
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.bn2 = nn.BatchNorm1d(hidden2_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        # 출력층
        self.fc3 = nn.Linear(hidden2_size, num_classes)

    def forward(self, x):
        '''
        순전파
        '''
        # 첫 번쨰 블록
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # 두 번째 블록
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # 출력층
        logits = self.fc3(x)
        return logits

# 모델 하이퍼파라미터
input_size = train_inputs.shape[1]  # 64
hidden1_size = 128
hidden2_size = 64
num_classes = 10  # 0~9
dropout_rate = 0.4

model = AdvancedClassifier(input_size, hidden1_size, 
                           hidden2_size, num_classes, dropout_rate).to(device)

print("\n" + "=" * 60)
print("모델 구조")
print("=" * 60)
print(model)
print("=" * 60)

# 파라미터 수 계산
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n총 파라미터 수: {total_params:,}")
print(f"학습 가능한 파라미터 수: {trainable_params:,}")

# 손실 함수 및 옵티마이저
# 손실 함수
criterion = nn.CrossEntropyLoss()

# 옵티마이저: Adam
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습 스케쥴러 : 검증 손실이 개선되지 않으면 학습률 감소
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',     # 손실을 최소화
    factor=0.5,     # 학습률을 0.5배로 감소
    patience=10     # 10에포크 동안 개선 없으면 감소  
)

print(f"손실 함수: {criterion}")
print(f"옵티마이저: {optimizer.__class__.__name__}")
print(f"초기 학습률: {learning_rate}")
print(f"학습률 스케줄러: ReduceLROnPlateau")

def evaluate(model, inputs, labels):
    '''
    모델 평가 함수

    Returns:
        tuple: (loss, accuracy, predictions)
    '''
    model.eval()
    dev = next(model.parameters()).device

    inputs=inputs.to(dev)
    labels=labels.to(dev)

    with torch.no_grad():
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs,1)
        # _ : 최대값 => 여기서는 인덱스만 필요하니깐 필요없어서 무시
        # predicted: 인덱스(클래스 번호)
        correct = (predicted==labels).sum().item()
        accuracy = correct / len(labels)

    return loss.item(), accuracy, predicted

# 학습 루프
def train_with_early_stopping(model, train_inputs, train_labels, val_inputs, val_labels,
                              num_epochs=300, patience=30, print_interval=20):
    '''
    Early Stopping을 포함한 학습 함수

    Args:
        model: 학습할 모델
        train_inputs, train_labels: 훈련 데이터
        val_inputs, val_labels: 검증 데이터
        num_epochs: 최대 에포크 수
        patience: Early Stopping을 위한 인내 에포크
        print_interval: 출력 간격

    Returns:
        dict: 학습 이력
    '''
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    print("\n" + "=" * 80)
    print(f"{'Epoch':^8} | {'Train Loss':^12} | {'Train Acc':^10} | {'Val Loss':^12} | {'Val Acc':^10} | {'Status':^12}")
    print("=" * 80)
   
    for epoch in range(num_epochs):
        # 학습 단계
        model.train()
        outputs = model(train_inputs)

        optimizer.zero_grad()
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()

        # 평가 단계
        # evaluate(): 모델을 eval 모드 + no_grad 상태로 두고 데이터셋에 대한
        # loss/accuracy 같은 성능을 계산하는 함수
        train_loss, train_acc, _ = evaluate(model, train_inputs, train_labels)
        val_loss, val_acc, _ = evaluate(model, val_inputs, val_labels)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # 학습률 스케줄러 업데이트
        scheduler.step(val_loss)

        # Early Stopping 체크
        status = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
            status = "✓ Best"
        else:
            epochs_no_improve +=1
            status =f'({epochs_no_improve}/{patience})'
        
        # 출력
        if (epoch + 1) % print_interval == 0 or epoch == 0:
            print(f"{epoch+1:^8} | {train_loss:^12.4f} | {train_acc:^10.4f} | {val_loss:^12.4f} | {val_acc:^10.4f} | {status:^12}")

        # Early Stopping
        if epochs_no_improve >= patience:
            print("=" * 80)
            print(f"Early Stopping at Epoch {epoch+1}")
            print(f"Best validation loss: {best_val_loss:.4f}")
            break
    print("=" * 80)
    print("학습 완료!")

    # 최고 성능 모델 복원
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"최고 성능 모델로 복원 (검증 손실: {best_val_loss:.4f})")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }

# 모델 학습
history = train_with_early_stopping(
    model,
    train_inputs, train_labels,
    val_inputs, val_labels,
    300,30,20
)

# 학습 과정 시각화
def plot_history(history):
    """
    학습 이력 시각화
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Loss 그래프
    axes[0].plot(history['train_losses'], label='Train Loss', linewidth=2.5, color='#3498db')
    axes[0].plot(history['val_losses'], label='Validation Loss', linewidth=2.5, color='#e74c3c')
    axes[0].set_xlabel('Epoch', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=13, fontweight='bold')
    axes[0].set_title('Training and Validation Loss', fontsize=15, fontweight='bold')
    axes[0].legend(fontsize=12, loc='upper right')
    axes[0].grid(True, alpha=0.3, linestyle='--')

    # Accuracy 그래프
    axes[1].plot(history['train_accs'], label='Train Accuracy', linewidth=2.5, color='#3498db')
    axes[1].plot(history['val_accs'], label='Validation Accuracy', linewidth=2.5, color='#e74c3c')
    axes[1].set_xlabel('Epoch', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    axes[1].set_title('Training and Validation Accuracy', fontsize=15, fontweight='bold')
    axes[1].legend(fontsize=12, loc='lower right')
    axes[1].grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.show()

# 시각화
plot_history(history)

# 테스트 데이터 평가
test_loss, test_acc, test_predictions = evaluate(model, test_inputs, test_labels)

print("\n" + "=" * 60)
print("최종 테스트 결과")
print("=" * 60)
print(f"테스트 손실: {test_loss:.4f}")
print(f"테스트 정확도: {test_acc:.4f} ({test_acc*100:.2f}%)")
print("=" * 60)

# 혼동행렬 시각화
'''
혼동 행렬이란?
    각 클래스별 예측 성능을 한눈에 파악할 수 있는 표
    행(row): 실제 레이블
    열(column): 예측 레이블
    대각선: 정확히 예측한 샘플 수
    대각선 외: 잘못 예측한 샘플 수
'''
def plot_confusion_matrix(true_labels, predicted_labels):
    '''
    혼동 행렬 시각화 함수

    Args:
        true_labels: 실제 레이블 (텐서)
        predicted_labels: 예측 레이블 (텐서)
    '''
    # 텐서를 NumPy로 변환
    y_true = true_labels.cpu().numpy()
    y_pred = predicted_labels.cpu().numpy()

    # 혼동 행렬 계산
    cm = confusion_matrix(y_true, y_pred)

    # 시각화
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,           # 숫자 표시
        fmt='d',              # 정수 형식
        cmap='Blues',         # 색상
        square=True,          # 정사각형 셀
        cbar_kws={'shrink': 0.8},
        linewidths=1,
        linecolor='gray'
    )

    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)

    # 축 레이블 설정
    plt.xticks(np.arange(10) + 0.5, range(10), fontsize=11)
    plt.yticks(np.arange(10) + 0.5, range(10), fontsize=11, rotation=0)

    plt.tight_layout()
    plt.show()

    # 클래스별 정확도 출력
    print("\n" + "=" * 60)
    print("클래스별 정확도")
    print("=" * 60)
    for i in range(10):
        class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"숫자 {i}: {class_acc:.4f} ({class_acc*100:.2f}%) - {cm[i, i]}/{cm[i].sum()}개")
    print("=" * 60)

# 혼동 행렬 시각화
plot_confusion_matrix(test_labels, test_predictions)

# 상세 성능 지표
'''
상세 성능 지표 (Classification Report)
    주요 지표 설명
        Precision (정밀도): 양성으로 예측한 것 중 실제 양성의 비율
            "이 숫자가 5라고 예측했을 때, 실제로 5일 확률"
            Precision = TP / (TP + FP)
        Recall (재현율): 실제 양성 중 양성으로 예측한 비율
            "실제 5인 숫자 중에서 5로 올바르게 예측한 비율"
            Recall = TP / (TP + FN)
        F1-Score: Precision과 Recall의 조화 평균
            F1 = 2 × (Precision × Recall) / (Precision + Recall)
Support: 각 클래스의 실제 샘플 수
'''

# 상세 성능 지표
def print_classification_report(true_labels, predicted_labels):
    '''
    상세 성능 보고서 출력
    '''
    y_true = true_labels.cpu().numpy()
    y_pred = predicted_labels.cpu().numpy()

    # Classification Report
    print("\n" + "=" * 70)
    print("상세 성능 보고서 (Classification Report)")
    print("=" * 70)
    print(classification_report(
        y_true,
        y_pred,
        target_names=[f'Digit {i}' for i in range(10)],
        digits=4
    ))

    # 전체 지표
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    print("=" * 70)
    print("전체 평균 지표 (Macro Average)")
    print("=" * 70)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("=" * 70)

# 성능 보고서 출력
print_classification_report(test_labels, test_predictions)

# 잘못 예측한 샘플 분석
def analyze_misclassifications(true_labels, predicted_labels, test_data, n_samples=10):
    '''
    잘못 예측한 샘플 분석 및 시각화

    Args:
        true_labels: 실제 레이블
        predicted_labels: 예측 레이블
        test_data: 테스트 입력 데이터 (표준화 전)
        n_samples: 표시할 샘플 수
    '''
    y_true = true_labels.cpu().numpy()
    y_pred = predicted_labels.cpu().numpy()

    # 잘못 예측한 인덱스 찾기
    misclassified_indices = np.where(y_true != y_pred)[0]

    print(f"\n총 {len(misclassified_indices)}개의 샘플이 잘못 예측되었습니다.")
    print(f"오분류율: {len(misclassified_indices)/len(y_true)*100:.2f}%\n")

    if len(misclassified_indices) == 0:
        print("완벽한 예측! 잘못 예측한 샘플이 없습니다.")
        return

    # 랜덤 샘플 선택
    sample_size = min(n_samples, len(misclassified_indices))
    sample_indices = np.random.choice(misclassified_indices, sample_size, replace=False)

    # 시각화
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()

    for i, idx in enumerate(sample_indices):
        # 이미지 데이터 복원 (8x8)
        image = test_data[idx].reshape(8, 8)

        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(
            f'True: {y_true[idx]} | Pred: {y_pred[idx]}',
            fontsize=11,
            color='red',
            fontweight='bold'
        )
        axes[i].axis('off')

    plt.suptitle('Misclassified Samples', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

# 오분류 샘플 분석
analyze_misclassifications(test_labels, test_predictions, X_test)

# 모델 저장 및 불러오기
def save_model(model, filepath='digit_classifier.pth'):
    '''
    모델을 파일로 저장

    Args:
        model: 저장할 모델
        filepath: 저장 경로
    '''
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': {
            'input_size': input_size,
            'hidden1_size': hidden1_size,
            'hidden2_size': hidden2_size,
            'num_classes': num_classes,
            'dropout_rate': dropout_rate
        }
    }, filepath)
    print(f"\n모델이 '{filepath}'에 저장되었습니다.")

def load_model(filepath='digit_classifier.pth'):
    '''
    저장된 모델 불러오기

    Args:
        filepath: 모델 파일 경로

    Returns:
        loaded_model: 불러온 모델
    '''
    checkpoint = torch.load(filepath, map_location=device)
    arch = checkpoint['model_architecture']

    # 모델 재생성
    loaded_model = AdvancedClassifier(
        arch['input_size'],
        arch['hidden1_size'],
        arch['hidden2_size'],
        arch['num_classes'],
        arch['dropout_rate']
    ).to(device)

    # 가중치 로드
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()

    print(f"\n모델이 '{filepath}'에서 불러와졌습니다.")
    return loaded_model

# 모델 저장
save_model(model, 'digit_classifier.pth')

# 모델 불러오기 테스트
loaded_model = load_model('digit_classifier.pth')

# 불러온 모델로 테스트
test_loss_loaded, test_acc_loaded, _ = evaluate(loaded_model, test_inputs, test_labels)
print(f"\n불러온 모델의 테스트 정확도: {test_acc_loaded:.4f} ({test_acc_loaded*100:.2f}%)")

# 실정 예측 함수
def predict_digit(model, scaler, digit_image):
    '''
    새로운 손글씨 숫자 이미지를 예측하는 함수

    Args:
        model: 학습된 모델
        scaler: 표준화에 사용한 scaler
        digit_image: 8x8 픽셀 이미지 (NumPy 배열)

    Returns:
        tuple: (predicted_digit, probabilities)
    '''
    dev = next(model.parameters()).device
    # 이미지를 1D 벡터로 변환
    digit_vector = digit_image.flatten().reshape(1, -1)

    # 표준화
    digit_scaled = scaler.transform(digit_vector)

    # 텐서 변환
    digit_tensor = torch.FloatTensor(digit_scaled).to(dev)

    # 예측
    model.eval()
    with torch.no_grad():
        output = model(digit_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_digit = torch.argmax(probabilities).item()

    return predicted_digit, probabilities.cpu().numpy()

# 테스트 샘플로 예측 테스트
sample_idx = np.random.randint(0, len(X_test))
sample_image = X_test[sample_idx].reshape(8, 8)
true_digit = y_test[sample_idx]

# 예측
predicted_digit, probs = predict_digit(model, scaler, sample_image)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 이미지 표시
axes[0].imshow(sample_image, cmap='gray')
axes[0].set_title(f'True: {true_digit} | Predicted: {predicted_digit}',
                  fontsize=13, fontweight='bold')
axes[0].axis('off')

# 확률 분포
axes[1].bar(range(10), probs, color='steelblue', alpha=0.7)
axes[1].set_xlabel('Digit', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Probability', fontsize=12, fontweight='bold')
axes[1].set_title('Prediction Probabilities', fontsize=13, fontweight='bold')
axes[1].set_xticks(range(10))
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("\n각 숫자별 예측 확률:")
for i, prob in enumerate(probs):
    print(f"  숫자 {i}: {prob:.4f} ({prob*100:.2f}%)")