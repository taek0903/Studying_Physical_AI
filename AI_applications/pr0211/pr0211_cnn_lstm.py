'''
이미지 캡셔닝: CNN + LSTM
캡션: The dog is running in the park

특징벡터(Feature Vector)
 * 이미지를 수자의 배열로 압축
  ** 원본 이미지 224 * 224 * 3 px
  ** 특징 벡터 256개 숫자로 압축
 * 강아지 사진 => [0.8.-0.3.1.2...]
 * 0.8: 귀, -0.3: 꼬리

토큰
실제 생성 과정
 * step 1: 이미지특징 -> '강아지가' 예측
 * step 2: '강아지가' + 이미지특징 -> '공원에서' 예측
 * step 3: '공원에서' + 이미지특징 -> '뛰어논다' 예측
 * step 4: '뛰어논다' + 이미지특징 -> 예측

LSTM에서 캡서닝(글) 생성
 * start → "강아지가" (70% 확률)
 * "강아지가" → "공원에서" (85% 확률)
 * "공원에서" → "즐겁게" (60% 확률)
 * "즐겁게" → "뛰어논다" (75% 확률)
 * "뛰어논다" → end (90% 확률)
최종결과: 강아지가 공원에서 즐겁게 뛰어논다.

[정리]
1. CNN: 이미지 에서 중요 특징 추출
2. LSTM 그 특징(벡터)의 가중치를 받아서 문장 생성
'''
# 모듈 가져오기

# 운영 체제와 상호 작용하는 함수(예: 파일 경로 처리)를 불러옴.
import os
# 자연어 처리를 위한 NLTK(Natural Language Toolkit) 라이브러리를 불러옴.
import nltk
# Python 객체를 직렬화(저장 및 불러오기)하기 위한 pickle 모듈을 불러옴.
import pickle
# NumPy 라이브러리를 불러옴.
import numpy as np
# PIL(Pillow) 라이브러리의 Image 모듈을 불러옴. 이미지 처리에 사용함.
from PIL import Image
# 요소의 빈도를 계산하는 Counter 클래스를 불러옴. (단어 빈도 계산에 사용)
from collections import Counter
# COCO 데이터셋 주석 파일을 다루기 위한 pycocotools의 COCO 객체를 불러옴.
from pycocotools.coco import COCO
# Matplotlib의 pyplot 모듈을 plt 별칭으로 불러옴. 시각화에 사용함.
import matplotlib.pyplot as plt
# PyTorch의 핵심 라이브러리를 불러옴.
import torch
# 신경망 레이어(nn) 모듈을 불러옴.
import torch.nn as nn
# 데이터셋 처리를 위한 데이터 유틸리티(data) 모듈을 불러옴.
import torch.utils.data as data
# TorchVision의 이미지 변환(transforms) 모듈을 불러옴.
from torchvision import transforms
# TorchVision의 미리 학습된 모델(models) 모듈을 불러옴. (특징 추출에 사용)
import torchvision.models as models
# 이미지 변환(transforms) 모듈을 다시 불러옴.
import torchvision.transforms as transforms
# RNN 배치 처리를 위한 pack_padded_sequence 유틸리티를 불러옴.
from torch.nn.utils.rnn import pack_padded_sequence
import urllib.request
import zipfile
import time
import pickle
from tqdm import tqdm

data_dir = r'D:\rokey\AI_applications\pr0211'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# urls = [
#     "http://images.cocodataset.org/zips/train2014.zip",
#     "http://images.cocodataset.org/zips/val2014.zip",
#     "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
# ]

# # 진행률 표시 함수
# def show_progress(block_num, block_size, total_size):
#     downloaded = block_num * block_size
#     if total_size > 0:
#         percent = downloaded * 100 / total_size
#         # 5% 단위로만 출력 (너무 자주 출력하면 느려짐)
#         if int(percent) % 5 == 0 and int(percent) != int((downloaded - block_size) * 100 / total_size):
#             print(f"{percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)")

# for url in urls:
#     filename = url.split('/')[-1]
#     file_path = os.path.join(data_dir, filename)

#     if not os.path.exists(file_path):
#         print(f'\nDownloading {filename}...')
#         # reporthook을 추가하여 진행률 표시
#         urllib.request.urlretrieve(url, file_path, reporthook=show_progress) 
#         print(f'\nDownload complete: {filename}')
#     else:
#         print(f'{filename} already exists. Skipping.')
    
#     print(f'Unzipping {filename}...')
#     try:
#         with zipfile.ZipFile(file_path, 'r') as zip_ref:
#             zip_ref.extractall(data_dir)
#         print(f'Unzip complete.')
        
#         # 원본 zip 삭제 (선택사항)
#         os.remove(file_path)
        
#     except Exception as e:
#         print(f"Error: {e}")

# print('All tasks finished')

nltk.download('punkt', download_dir=data_dir)
nltk.download('punkt_tab', download_dir=data_dir)
nltk.data.path.append(data_dir)
class Vocab(object):
    def __init__(self):
        self.w2i = {}
        self.i2w = {}
        self.index = 0

    def __call__(self, token):
        if not token in self.w2i:
            return self.w2i['<unk>']
        return self.w2i[token]
    
    def __len__(self):
        return len(self.w2i)
    
    def add_token(self, token):
        if not token in self.w2i:
            self.w2i[token] = self.index
            self.i2w[self.index] = token
            self.index +=1

def build_vocabulary(json, threshold):
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()

    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if (i+1) % 1000 == 0:
            print(f"[{i+1}/{len(ids)}] Tokenized the captions.")

    tokens = [token for token, cnt in counter.items() if cnt >= threshold]

    vocab = Vocab()
    vocab.add_token('<pad>')
    vocab.add_token('<start>')
    vocab.add_token('<end>')
    vocab.add_token('<unk>')

    for i, token in enumerate(tokens):
        vocab.add_token(token)

    return vocab

# 1. 프로젝트 기본 경로 (사용자 환경)
base_dir = r'D:\rokey\AI_applications\pr0211'

# 2. 입력 파일 경로 (JSON)
# 압축을 풀면 'data_dir/annotations/...' 위치에 생깁니다.
json_path = os.path.join(base_dir,  'annotations', 'captions_train2014.json')

# 3. 출력 파일 경로 (만들어진 단어장을 저장할 곳)
vocab_path = os.path.join(base_dir, 'vocabulary.pkl')

# 4. 실행 및 저장
print(f"단어장 생성을 시작합니다.\n입력 파일: {json_path}")

# build_vocabulary 함수가 정의되어 있어야 합니다!
vocab = build_vocabulary(json=json_path, threshold=4)

# 만든 단어장을 파일로 저장 (pickle)
with open(vocab_path, 'wb') as f:
    pickle.dump(vocab, f)

print(f"단어장 저장 완료: {vocab_path}")

def reshape_image(image, shape):
    return image.resize(shape)

def reshape_images(image_path, output_path, shape):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    images = os.listdir(image_path)
    num_im = len(images)

    for i, im in enumerate(images):
        in_file = os.path.join(image_path, im)          
        out_file = os.path.join(output_path, im)        

        with Image.open(in_file) as image:              
            image = reshape_image(image, shape)
            image.save(out_file)

        if (i+1) % 100 == 0:
            print ("[{}/{}] Resized the images and saved into '{}'."
                    .format(i+1, num_im, output_path))

image_path = r'D:\rokey\AI_applications\pr0211\train2014'
output_path = r'D:\rokey\AI_applications\pr0211\resized_images'
image_shape = [256, 256]
# reshape_images(image_path,output_path, image_shape)

class CustomCocoDataset(data.Dataset):
    def __init__(self, data_path, coco_json_path, vocabulary, transform=None):
        self.root = data_path
        self.coco_data = COCO(coco_json_path)
        self.indices = list(self.coco_data.anns.keys())
        self.vocabulary = vocabulary
        self.transform = transform

    def __getitem__(self, idx):
        coco_data = self.coco_data
        vocabulary = self.vocabulary
        annotations_id = self.indices[idx]
        caption = coco_data.anns[annotations_id]['caption']
        image_id = coco_data.anns[annotations_id]['image_id']
        image_path = coco_data.loadImgs(image_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, image_path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        word_tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocabulary('<start>'))
        caption.extend([vocabulary(token) for token in word_tokens])
        ground_truth = torch.Tensor(caption)

        return image, ground_truth
    
    def __len__(self):
        return len(self.indices)

def collate_function(data_batch):
    data_batch.sort(key=lambda d: len(d[1]), reverse=True)
    imgs, caps = zip(*data_batch)

    imgs = torch.stack(imgs, 0)

    cap_lens = [len(cap) for cap in caps]
    tgts = torch.zeros(len(caps), max(cap_lens)).long()

    for i, cap in enumerate(caps):
        end = cap_lens[i]
        tgts[i, :end] = cap[:end]

    return imgs, tgts, cap_lens

def get_loader(data_path, coco_json_path, vocavulary, transform, batch_size, shuffle, num_workers):
    coco_dataset = CustomCocoDataset(
        data_path=data_path,
        coco_json_path=coco_json_path,
        vocabulary=vocavulary,
        transform=transform
    )

    custom_data_loader = torch.utils.data.DataLoader(
        dataset=coco_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_function
    )

    return custom_data_loader

class CNNModel(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        resnet = models.resnet152(pretrained=True)
        module_list = list(resnet.children())[:-1]
        self.resnet_module = nn.Sequential(*module_list)
        self.linear_layer = nn.Linear(resnet.fc.in_features, embedding_size)
        self.batch_norm = nn.BatchNorm1d(embedding_size, momentum=0.01)

    def forward(self, input_images):
        with torch.no_grad():
            resnet_features = self.resnet_module(input_images)

        resnet_features =resnet_features.reshape(resnet_features.size(0), -1)
        final_features = self.batch_norm(self.linear_layer(resnet_features))
        return final_features
    
class LSTMModel(nn.Module):
    def __init__(self, embedding_size, hidden_layer_size, vocabulary_size, num_layers, max_seq_len=20):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocabulary_size, embedding_size)
        self.lstm_layer = nn.LSTM(embedding_size, hidden_layer_size, num_layers, batch_first=True)
        self.Linear_layer = nn.Linear(hidden_layer_size, vocabulary_size)
        self.max_seq_len = max_seq_len

    def forward(self, input_features, capts, lens):
        embeddings = self.embedding_layer(capts)
        embeddings = torch.cat((input_features.unsqueeze(1), embeddings), 1)
        lstm_input = pack_padded_sequence(embeddings, lens, batch_first=True)
        hidden_variables, _ = self.lstm_layer(lstm_input)
        model_outptuts = self.Linear_layer(hidden_variables[0])
        return model_outptuts
    
    def sample(self, input_features, lstm_states=None):
        lstm_inputs = input_features.unsqueeze(1)
        sampled_indices = []
        for i in range(self.max_seq_len):
            hidden_variables, lstm_states = self.lstm_layer(lstm_inputs, lstm_states)
            model_outputs = self.Linear_layer(hidden_variables.squeeze(1))
            _, predicted_outputs = model_outputs.max(1)
            sampled_indices.append(predicted_outputs)

            lstm_inputs = self.embedding_layer(predicted_outputs)
            lstm_inputs = lstm_inputs.unsqueeze(1)

        sampled_indices = torch.stack(sampled_indices, 1)
        return sampled_indices
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_path = os.path.join(data_dir, 'vocabulary.pkl')

image_dir = os.path.join(data_dir, 'train2014')
caption_path = os.path.join(data_dir, 'annotations', 'captions_train2014.json')
models_dir = os.path.join(data_dir, 'models_dir')

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

with open(vocab_path, 'rb') as f:
    vocabulary = pickle.load(f)

# ResNet 사전 학습에 사용된 표준에 맞춘 이미지 전처리(정규화 포함)를 정의함.
transform = transforms.Compose([
    transforms.Resize(256),
    # 무작위 자르기를 적용함.
    transforms.RandomCrop(224),
    # 무작위 수평 뒤집기를 적용함. (데이터 증강)
    transforms.RandomHorizontalFlip(),
    # 이미지를 텐서로 변환함. >> [0,1]
    transforms.ToTensor(),
    # ResNet 사전 학습 시 사용된 평균 및 표준편차로 정규화함.
    transforms.Normalize((0.485, 0.456, 0.406),
                          (0.229, 0.224, 0.225))
    ])

custom_data_loader = get_loader(
    image_dir,              # 이미지 폴더 경로
    caption_path,           # 캡션 json 파일 경로
    vocabulary,            # 위에서 불러온 단어장
    transform,         # (이미 정의되어 있어야 함) 이미지 전처리
    128,              # 한 번에 학습할 데이터 수
    shuffle=True,                # 데이터를 섞어서 학습
    num_workers=0                # ★중요: 윈도우에서는 0으로 해야 에러가 안 납니다!
)

# CNN 인코더 모델을 구축하고 장치(device)로 이동시킴. 출력 임베딩 크기는 256임.
encoder_model = CNNModel(256).to(device)
# LSTM 디코더 모델을 구축하고 장치로 이동시킴. (임베딩 256, 은닉층 512, 어휘 크기 len(vocabulary), 레이어 1개)
decoder_model = LSTMModel(256, 512, len(vocabulary), 1).to(device)

# 손실함수, 옵티마이저 설정
# 손실함수, cross entropy loss 정의 (단어 예측을 위한 분류 손실)
loss_criterion = nn.CrossEntropyLoss()

parameters=\
list(decoder_model.parameters()) + list(encoder_model.linear_layer.parameters()) + list(encoder_model.batch_norm.parameters())

optimizer = torch.optim.Adam(parameters, lr=0.001)

# 모델 학습 루프
total_num_steps = len(custom_data_loader)

for epoch in range(1):
    loop = tqdm(custom_data_loader, desc=f'Epoch {epoch+1}/1')
    for i, (imgs, caps, lens) in enumerate(loop):
        imgs = imgs.to(device)
        caps = caps.to(device)

        tgts = pack_padded_sequence(caps, lens, batch_first=True)[0]

        # 순전파 역전파 최적화

        feats = encoder_model(imgs)
        outputs = decoder_model(feats, caps, lens)

        loss = loss_criterion(outputs, tgts)

        decoder_model.zero_grad()
        encoder_model.zero_grad()

        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item(), perplexity=np.exp(loss.item()))

        if i % 10 == 0:
            # 손실 값과 함께, 복잡도(Perplexity)를 계산하여 출력함 (exp(loss)).
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                  .format(epoch, 5, i, total_num_steps, loss.item(), np.exp(loss.item())))

        # 1000 스텝마다 모델 가중치를 체크포인트로 저장함.
        if (i+1) % 1000 == 0:
            # 디코더 모델의 가중치를 파일로 저장함.
            torch.save(decoder_model.state_dict(), os.path.join(
                'models_dir/', 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
            # 인코더 모델의 가중치를 파일로 저장함.
            torch.save(encoder_model.state_dict(), os.path.join(
                'models_dir/', 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))

# 캡션을 생성할 이미지 파일 경로를 정의함.
image_file_path = r'D:\rokey\AI_applications\data\cat_on_a_laptop.jpg'

# 장치(Device) 설정을 정의함.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 이미지 로딩 및 크기 조정 함수 ---

# 이미지 파일을 로드하고 전처리하는 함수를 정의함.
def load_image(image_file_path, transform=None):
    # 이미지를 열고 RGB 형식으로 변환함.
    img = Image.open(image_file_path).convert('RGB')
    # 이미지 크기를 [224, 224]로 조절함. LANCZOS는 고품질 리샘플링 필터임.
    img = img.resize([224, 224], Image.LANCZOS)

    # 변환 함수(transform)가 있으면 적용하고 배치 차원(unsqueeze(0))을 추가함.
    if transform is not None:
        img = transform(img).unsqueeze(0)

    # 텐서를 반환함.
    return img

with open(vocab_path, 'rb') as f:
    vocabulary = pickle.load(f)

# CNN 인코더 모델을 구축하고 평가 모드(eval())로 설정함.
encoder_model = CNNModel(256).eval()
# LSTM 디코더 모델을 구축함.
decoder_model = LSTMModel(256, 512, len(vocabulary), 1)

# 모델들을 정의된 장치로 이동시킴.
encoder_model = encoder_model.to(device)
decoder_model = decoder_model.to(device)

models_dir = os.path.join(data_dir, 'models_dir')
# 미리 학습된 모델 가중치 파일(체크포인트)을 로드함.
encoder_path = os.path.join(models_dir, 'encoder-1-3000.ckpt')
decoder_path = os.path.join(models_dir, 'decoder-1-3000.ckpt')

img = load_image(image_file_path, transform)
# 이미지 텐서를 장치로 이동시킴.
img_tensor = img.to(device)


# 인코더에 이미지 텐서를 통과시켜 특징 벡터를 얻음.
feat = encoder_model(img_tensor)
# 디코더의 sample 함수를 사용하여 특징 벡터로부터 캡션 인덱스 시퀀스를 생성함. (Greedy Search)
sampled_indices = decoder_model.sample(feat)
# 생성된 인덱스 텐서를 CPU로 이동시키고 NumPy 배열로 변환한 후, 배치 차원(0)을 제거함.
sampled_indices = sampled_indices[0].cpu().numpy()


# 단어 인덱스 시퀀스를 실제 단어 문자열로 변환함.
predicted_caption = []
for token_index in sampled_indices:
    # 인덱스에 해당하는 단어를 어휘 집합에서 찾음.
    word = vocabulary.i2w[token_index]
    # 단어를 리스트에 추가함.
    predicted_caption.append(word)
    # 문장 종료 토큰('<end>')을 만나면 변환을 중단함.
    if word == '<end>':
        break
# 단어 리스트를 공백으로 연결하여 최종 문장을 만듦.
predicted_sentence = ' '.join(predicted_caption)


# --- 결과 출력 및 시각화 ---

# 생성된 캡션 문장을 출력함.
print (predicted_sentence)
# 원본 이미지를 다시 로드함.
img = Image.open(image_file_path)
# 원본 이미지를 화면에 표시함.
plt.imshow(np.asarray(img))