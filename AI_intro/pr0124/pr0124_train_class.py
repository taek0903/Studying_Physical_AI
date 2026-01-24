import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchinfo import summary
from torchviz import make_dot
from tqdm import tqdm
import torchvision.datasets as datasets
from sklearn.datasets import load_digits  # 손글씨 숫자 데이터 (0~9)
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
import seaborn as sns
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve
)
import json

plt.rcParams["font.family"] = "Malgun Gothic"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

np.random.seed(42)
torch.manual_seed(42)

class Config:
    def __init__(self):
        self.test_size = 0.2                # 테스트 데이터 비율
        self.val_size = 0.2                 # 검증 데이터 비율
        self.random_state = 42              # 랜덤 시드 설정
        self.input_dim = 10                 # 들어가는 차원 수
        self.hidden_dim = [64,32,16]        # 은닉층 layer 크기
        self.dropout_rate = 0.3             # 드롭아웃 비율
        self.batch_size = 32                # 배치 사이즈
        self.num_epochs = 200               # 학습 반복 횟수
        self.learning_rate = 0.001          # 학습률
        self.weight_decay = 0.0001          # 과적합 방지용 벌점 세기
        self.patience = 20                  # earliy Stopping 참아 주는 정도
        self.min_delta = 0.001              # 검증 성능이 좋아졌다고 인정하는 최소 기준선
        self.scheduler_step_size = 30       # 브레이크를 언제 밟을지
        self.scheduler_gamma = 0.5          # 브레이크를 얼마나 세게 밟을지

config = Config()
print('Config 생성완료!')
print(f'Batch Size: {config.batch_size}')
print(f'Learning Rate: {config.learning_rate}')

# 데이터 전처리
class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()

    def load_and_prepare_data(self):
        diabetes = load_diabetes()
        X = diabetes.data
        y_regression = diabetes.target
        median = np.median(y_regression)
        y = (y_regression > median).astype(int)

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.config.test_size,
            stratify=y, random_state=self.config.random_state
        )

        # Train = Test 분류
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=self.config.val_size,
            stratify=y_train_val, random_state=self.config.random_state
        )

        # Train data set 을 Train - Validation 분류

        X_train = self.scaler.fit_transform(X_train)
        # fit_transform(X_train)훈련하니깐 fit_transform
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        print('데이터 준비 완료')
        print(f'Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}')

        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_dataloader(self, X_train, X_val, X_test, y_train, y_val, y_test):
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train).view(-1,1)
        )

        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val).view(-1,1)
        )

        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(y_test).view(-1,1)
        )

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

        print(f'DataLoader 생성 완료 (Batch Size: {self.config.batch_size})')

        return train_loader, val_loader, test_loader

print('DataPreprocessor 클래스 정의 완료!')

preprocessor = DataPreprocessor(config)
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.load_and_prepare_data()
train_loader, val_loader, test_loader = preprocessor.create_dataloader(
    X_train, X_val, X_test, y_train, y_val, y_test
)

# 모델 학습 과정
class DiabetesClassifier(nn.Module):    
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)
    
model = DiabetesClassifier(
    input_dim=config.input_dim,
    hidden_dims=config.hidden_dim,
    dropout_rate=config.dropout_rate
)
model = model.to(device)

print('모델 생성완료')
print(model)

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_State = model.state_dict()
            self.counter = 0

print('EarlyStopping 클래스 정의 완료!')

# 학습기
class Trainer:
    def __init__(self, model, config, device='cpu'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr = config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.scheduler_step_size,
            gamma = config.scheduler_gamma
        )
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta
        )
        self.history = {
            'train_loss' : [],
            'val_loss' : [],
            'train_acc' : [],
            'val_acc' : [],
            'learning_rate' : []
        }

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch_X.size(0)
            predicted = (outputs >= 0.0).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        return total_loss / total, correct / total
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

                total_loss += loss.item() * batch_X.size(0)
                predicted = (outputs >= 0.0).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        return total_loss / total, correct / total
    
    def fit(self, train_loader, val_loader):
        print('학습 시작...')

        for epoch in range(self.config.num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1:3d} / {self.config.num_epochs}] '
                      f'Train [{train_loss:.4f} | Train Acc: {train_acc:.4f}] | '
                      f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | '
                      f'LR: {current_lr:.6f}')
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print(f'Early Stopping at Epoch {epoch+1}')
                self.model.load_state_dict(self.early_stopping.best_model_State)
                break

        print('\n 학습 완료!')
print('Trainer 클래스 정의 완료!')

trainer = Trainer(model, config)
trainer.fit(train_loader, val_loader)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

axes[0, 0].plot(trainer.history['train_loss'], label='Train Loss')
axes[0, 0].plot(trainer.history['val_loss'], label='Val Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Loss Curve')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(trainer.history['train_acc'], label='Train Acc')
axes[0, 1].plot(trainer.history['val_acc'], label='Val Acc')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Accuracy Curve')
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[1, 0].plot(trainer.history['learning_rate'], color='orange')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Learning Rate')
axes[1, 0].set_title('Learning Rate Schedule')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(True)

axes[1, 1].plot(trainer.history['train_loss'], label='Train', alpha=0.7)
axes[1, 1].plot(trainer.history['val_loss'], label='Val', alpha=0.7)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].set_title('Train vs Val Loss')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            probs = torch.sigmoid(outputs)
            preds = (outputs >= 0.0).float()

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    all_preds = np.array(all_preds).flatten()
    all_probs = np.array(all_probs).flatten()
    all_labels = np.array(all_labels).flatten()

    return all_labels, all_preds, all_probs

all_labels, all_preds, all_probs = evaluate_model(model, test_loader)

print('[Classification Report]')
print(classification_report(all_labels, all_preds, target_names=['Low Risk', 'High Risk']))

cm = confusion_matrix(all_labels, all_preds)
print('\n[Confusion Matrix]')
print(f'              Predicted')
print(f'            Low  High')
print(f'Actual Low  {cm[0,0]:3d}  {cm[0,1]:3d}')
print(f'       High {cm[1,0]:3d}  {cm[1,1]:3d}')

auc = roc_auc_score(all_labels, all_probs)
print(f'\nAUC: {auc:.4f}')

precision, recall, _ = precision_recall_curve(all_labels, all_probs)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, linewidth=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.show()