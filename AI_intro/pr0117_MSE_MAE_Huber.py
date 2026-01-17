"""
Part 1: 회귀 손실함수 비교
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- Huber Loss

독립적으로 실행 가능합니다.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(42)
np.random.seed(42)

print("=" * 70)
print("Part 1: 회귀 손실함수 비교")
print('=' * 70)

# 실습 1-1 : 손실함수의 동작 원리 이해
print("\n[실습 1-1] 손실함수 동작 원리")

y_true = torch.tensor([10.0, 20.0, 30.0])
y_pred = torch.tensor([12.0, 19.0, 35.0])

mse_loss = nn.MSELoss()
mse_value = mse_loss(y_pred, y_true)

mae_loss = nn.L1Loss()
mae_value = mae_loss(y_pred, y_true)

huber_loss = nn.HuberLoss(delta=1.0)        
huber_value = huber_loss(y_pred, y_true)    
# 이상치에 덜 민감하면서도 학습은 안정적  
# delta 값을 기준으로 MSE 혹은 MAE 선택
# 오차가 작을 땐 MSE 오차가 클 땐 MAE                                          

print(f'예측값: {y_pred.numpy()}')
print(f'실제값: {y_true.numpy()}')
print(f'오차: {(y_pred - y_true).numpy()}')

erros = (y_pred-y_true).numpy()
print(f'수동계산: {np.mean(erros**2):.4f}')
print(f'MSE: {np.mean(np.abs(erros)):.4f}')

# 실습 1-2: 이상치가 있을 때 
print("\n" + "=" * 70)
print("[실습 1-2] 이상치에 대한 민감도 비교")
print("=" * 70)

y_true_with_outlier = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
y_pred_normal = torch.tensor([11.0, 19.0, 31.0, 39.0, 51.0])
y_pred_with_outlier = torch.tensor([11.0, 19.0, 31.0, 100.0, 51.0])

print("\n정상 예측 (오차 모두 작음):")
print("실제:", y_true_with_outlier.numpy())
print("예측:", y_pred_normal.numpy())

mse_normal = mse_loss(y_pred_normal, y_true_with_outlier)
mae_normal = mae_loss(y_pred_normal, y_true_with_outlier)
huber_normal = huber_loss(y_pred_normal, y_true_with_outlier)

print(f'\nMSE: {mse_normal.item():.4f}')
print(f'MAE: {mae_normal.item():.4f}')
print(f'Huber: {huber_normal.item():.4f}')

print('\n이상치 포함 예측 (하나의 큰 오차):')
print(f'실제: {y_true_with_outlier.numpy()}')
print(f'예측: {y_pred_with_outlier.numpy()}')
print(f'오차: {(y_pred_with_outlier-y_true_with_outlier).numpy()}')

mse_outlier = mse_loss(y_pred_with_outlier, y_true_with_outlier)
mae_outlier = mae_loss(y_pred_with_outlier, y_true_with_outlier)
huber_outlier = huber_loss(y_pred_with_outlier, y_true_with_outlier)

print(f'\nMSE: {mse_outlier.item():.4f} (증가율: {mse_outlier/mse_normal:.1f})')
print(f'MAE: {mae_outlier.item():.4f} (증가율: {mae_outlier/mae_normal:.1f})')
print(f'Huber: {huber_outlier.item():.4f} (증가율: {huber_outlier/huber_outlier:.1f})')

print('\분석:')
print('MSE는 이상치에 매우민감 (제곱 때문)')
print('MAE는 이상치에 강건 (절대값만 고려)')
print('Huber는 중간 (작은 오차는 제곱, 큰 오차는 선형)')

print("\n" + "=" * 70)
print("[실습 1-3] 실제 회귀 문제에서 손실함수 비교")
print("=" * 70)

print("\n데이터 생성 중...")
X, y = make_regression(n_samples=500, n_features=10, noise=10.0, random_state=42)

n_outliters = int(0.1*len(y))
outliter_indices = np.random.choice(len(y), n_outliters, replace=False)
y[outliter_indices] += np.random.randn(n_outliters)*50

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # 학습 데이터로 전처리 기준을 만들고 기준으로 데이터 변환
X_test = scaler.transform(X_test)

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1) # 결과 데이터는 차원이 다르므로 차원을 맞춰줌
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

print(f'훈련 데이터: {X_train.shape}, 이상치: {n_outliters}개')

class RegressionModel(nn.Module):
    '''간단한 회귀 신경망'''
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(10,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1),
        )
    
    def forward(self, x):
        return self.network(x)
    
loss_functions = {
    'MSE' : nn.MSELoss(),
    'MAE' : nn.L1Loss(),
    'Huber' : nn.HuberLoss()
}

results = {}

for loss_name, criterion in loss_functions.items():
    print(f'\n{loss_name}로 학습중...')

    model = RegressionModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_losses = []
    num_epochs = 100

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t)

        test_mse = nn.MSELoss()(test_pred, y_test_t).item()
        test_mae = nn.L1Loss()(test_pred, y_test_t).item()
        test_huber = nn.HuberLoss()(test_pred, y_test_t).item()

    results[loss_name] = {
        'train_losses' : train_losses,
        'test_mse' : test_mse,
        'test_mae' : test_mae,
        'test_huber' : test_huber,
        'predictions' : test_pred
    }

    print(f'최종 훈련 손실: {train_losses[-1]:4f}')
    print(f'테스트 MSE: {test_mse:.4f}')
    print(f'테스트 MAE: {test_mae:.4f}')

print("\n" + "=" * 70)
print("[실습 1-4] 결과 시각화")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 학습 곡선 비교
ax1 = axes[0, 0]
colors = ['#e74c3c', '#3498db', '#2ecc71']
for (loss_name, result), color in zip(results.items(), colors):
    ax1.plot(result['train_losses'], label=loss_name, linewidth=2, color=color)

ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Training Loss', fontsize=11)
ax1.set_title('Training Loss Comparison', fontsize=12, weight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

# 2. 테스트 성능 비교 (MAE 기준)
ax2 = axes[0, 1]
loss_names = list(results.keys())
test_maes = [results[name]['test_mae'] for name in loss_names]

bars = ax2.bar(loss_names, test_maes, color=colors, edgecolor='black', alpha=0.7)
ax2.set_ylabel('Test MAE', fontsize=11)
ax2.set_title('Test MAE Comparison', fontsize=12, weight='bold')
ax2.grid(axis='y', alpha=0.3)

# 값 표시
for bar, mae in zip(bars, test_maes):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{mae:.2f}', ha='center', va='bottom', fontsize=10, weight='bold')

# 3. 예측 vs 실제 (MAE 모델)
ax3 = axes[1, 0]
mae_predictions = results['MAE']['predictions'].numpy().flatten()
ax3.scatter(y_test, mae_predictions, alpha=0.6, edgecolors='black', linewidths=0.5)
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Perfect Prediction')
ax3.set_xlabel('True Values', fontsize=11)
ax3.set_ylabel('Predicted Values', fontsize=11)
ax3.set_title('MAE Model: Prediction vs True', fontsize=12, weight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. 오차 분포 비교
ax4 = axes[1, 1]
for loss_name, color in zip(loss_names, colors):
    predictions = results[loss_name]['predictions'].numpy().flatten()
    errors = y_test - predictions
    ax4.hist(errors, bins=30, alpha=0.5, label=loss_name, color=color)

ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax4.set_xlabel('Prediction Error', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.set_title('Error Distribution', fontsize=12, weight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('part1_regression_loss_comparison.png', dpi=150, bbox_inches='tight')
print("\n저장: part1_regression_loss_comparison.png")
plt.close()

print("\n" + "=" * 70)
print("[실습 1-5] Huber Loss의 delta 파라미터 영향")
print("=" * 70)

delta_values = [0.5, 1.0, 2.0, 5.0]
errors = torch.linspace(-10, 10, 200)

plt.figure(figsize=(12,5))

for delta in delta_values:
    huber = nn.HuberLoss(delta=delta, reduction='none')
    loss_values = huber(errors, torch.zeros_like(errors))
    plt.plot(errors.numpy(), loss_values.numpy(),
             label = f'delta={delta}', linewidth=2)

mse_values = 0.5 * errors**2
mae_values = torch.abs(errors)

plt.plot(errors.numpy(), mse_values.numpy(),
         'k--', linewidth=2, label='MSE', alpha=0.5)
plt.plot(errors.numpy(), mae_values.numpy(),
         'k:', linewidth=2, label='MAE', alpha=0.5)

plt.xlabel('Prediction Error', fontsize=12)
plt.ylabel('Loss Value', fontsize=12)
plt.title('Huber Loss with Different Delta Values', fontsize=14, weight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.xlim(-10, 10)
plt.ylim(0, 50)

plt.tight_layout()
plt.savefig('part1_huber_delta_effect.png', dpi=150, bbox_inches='tight')
print("\n저장: part1_huber_delta_effect.png")
plt.close()

print("\n분석:")
print("  delta가 작을수록 MSE에 가까움 (작은 오차에 민감)")
print("  delta가 클수록 MAE에 가까움 (큰 오차에 관대)")
print("  delta=1.0이 일반적으로 좋은 기본값")

print("\n" + "=" * 70)
print("Part 1 완료")
print("=" * 70)

print("\n핵심 개념:")
print("\n1. MSE (Mean Squared Error)")
print("   - 큰 오차에 큰 페널티")
print("   - 이상치에 매우 민감")
print("   - 가장 널리 사용")

print("\n2. MAE (Mean Absolute Error)")
print("   - 모든 오차를 동등하게 취급")
print("   - 이상치에 강건")
print("   - 해석이 직관적")

print("\n3. Huber Loss")
print("   - MSE + MAE의 장점 결합")
print("   - delta로 경계 조절")
print("   - 로봇공학, 강화학습에서 선호")

print("\n실전 가이드:")
print("  - 일반적인 경우: MSE")
print("  - 이상치 많음: MAE 또는 Huber")
print("  - 빠른 수렴 필요: MSE")
print("  - 강건성 필요: Huber (delta=1.0)")

print("\n생성된 파일:")
print("  1. part1_regression_loss_comparison.png - 손실함수 비교")
print("  2. part1_huber_delta_effect.png - Huber delta 영향")

print("\n다음: Part 2 - 분류 손실함수")
print("=" * 70)