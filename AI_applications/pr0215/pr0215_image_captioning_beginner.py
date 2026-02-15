import os
import re
import zipfile
import random
from collections import Counter
import urllib.request
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# 재현성 위해서 seed 값 설정
def set_seed(seed: int = 42):       # seed 값을 받아서 여러 라이브러리의 난수 발생기를 고정하는 함수 정의
    random.seed(seed)               # 파이썬 기본 random 모듈의 시드를 고정
    np.random.seed(seed)            # 넘파이의 난수 시드를 고정
    torch.manual_seed(seed)         # PyTorch CPU 난수 시드를 고정
    if torch.cuda.is_available():   # 만약 GPU(CUDA)가 사용 가능하다면
        torch.cuda.manual_seed_all(seed)    # 모든 GPU의 난수 시드를 고정

set_seed(42)

# 전체 데이터셋이 들어갈 기본 폴더
data_dir = r'D:\rokey\AI_applications\pr0215'
os.makedirs(data_dir, exist_ok=True)

images_zip_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
text_zip_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"

images_zip_path = os.path.join(data_dir, "Flickr8k_Dataset.zip")
text_zip_path = os.path.join(data_dir, "Flickr8k_text.zip")

def download_if_not_exists(url, save_path):
    '''파일이 없을 때만 다운로드'''
    if not os.path.exists(save_path):
        print(f'다운로드 중:{url}')
        urllib.request.urlretrieve(url, save_path)
        print(f'완료: {save_path}')
    else:
        print(f'이미 존재: {save_path}')

download_if_not_exists(images_zip_url, images_zip_path)
download_if_not_exists(text_zip_url, text_zip_path)

zip_path = r'D:\rokey\AI_applications\pr0215\Flickr8k_Dataset.zip'
extract_path = r'D:\rokey\AI_applications\pr0215'

# print('압축해제 중...')
# with zipfile.ZipFile(zip_path, "r") as zf:
#     zf.extractall(extract_path)

# print('완료!')
# print('압축 해제 후 폴도 목록', os.listdir(extract_path))

'''
캡션 파일 로드 및 구조 이해
Flickr8k 파일에는 이미지 파일 이름과 그 이미지에 대한 여러 문장(캡션)이 함께 들어있음
예시 형식:
1000268201_693b08cb0e.jpg#0\tA child in a pink dress is climbing up a set of stairs in an entry way .
 * 1000268201_693b08cb0e.jpg : 이미지 파일 이름
 * #0 : 이 이미지에 대한 0번째 캡션 (한 이미지당 5개의 캡션)
 * 그 뒤 : 실제 문장
'''

'''
ex) data_dir이 'flick8k'의 상위 디렉토리라고 가정
사용자 환경에 맞춰 'data_dir' 변수 설정이 필요
만약 'dat_dir'이 이미 'flickr8k' 폴더를 가리킨다면 아래 코드는 필요 없음
여기서 'flickr8k' 폴더 안에 데이터가 있다고 가정하고 경로를 설정
**!!! 중요: 실제 환경에 맞게 data_dir값 설정!!!**

# --- 4. 캡션 파일 로드 ---
# 'Flickr8k_text.zip' 압축을 푼 후, 'Flickr8k.token.txt' 파일의 실제 경로를 확인하여 수정해야 함
# 일반적으로 'data_dir' 안에 압축을 풀면 'Flickr8k.token.txt' 파일이 바로 생김
# 만약 에러가 발생했다면, 파일 이름이 잘못되었거나 경로가 잘못되었을 가능성이 큼.

# 1. 파일 이름이 잘못되었을 경우 (혹시나 하여 오타 수정 가능성 포함):
# captions_file = os.path.join(data_dir, "Flickr8k.token.txt") # 기존 코드

# 2. 파일이 'flickr8k' 폴더 안에 있고, data_dir이 'flickr8k'의 상위 폴더인 경우:
# data_dir이 'flickr8k' 폴더를 포함하는 상위 경로일 경우 아래처럼 수정
# captions_file = os.path.join(data_dir, "flickr8k", "Flickr8k.token.txt")
# (이 경우 'data_dir'의 정확한 정의가 필요합니다.)

# ***가장 일반적인 해결책: 파일 이름이 정확하다면, 압축을 풀지 않았거나 파일의 실제 위치가 다른 것***

# **Flickr8k.token.txt 파일을 찾을 수 있는 올바른 경로로 수정**
# (예시: data_dir이 데이터를 담고 있는 최상위 폴더이고, 그 안에 'Flickr8k.token.txt'가 있다고 가정)
'''
zip_path_text = os.path.join(data_dir, "Flickr8k_text.zip")
extract_to_dir = data_dir

with zipfile.ZipFile(zip_path_text, 'r') as zip_ref:
    zip_ref.extractall(extract_to_dir)
    print(f"'{zip_path_text}' 파일이 '{extract_to_dir}'에 성공적으로 압축 해제되었습니다.")

captions_file = os.path.join(data_dir, 'Flicker8k.token.txt')    # 압축을 푼 파일 경로 지정

print('캡션 파일 경로', captions_file)
captions_file = r'D:\rokey\AI_applications\pr0215\Flickr8k.token.txt'
try:
    with open(captions_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print("전체 캡션 라인 수:", len(lines))
    print("앞에서 3줄만 미리 보기:")
    for i in range(3):
        print(lines[i].strip())

except Exception as e:
    print(f"\n파일을 읽는 중 예기치 않은 에러 발생: {e}")

'''
텍스트 전처리 및 이미지-캡션 매핑 만들기
이미지 파일 이름별로 여러 개의 캡션 문장을 모아 두기 위해, 다음과 같은 과정을 거침
 1. 한 줄씩 읽어 이미지 이름과 문장 부분 분리
 2. 문장의 불필요한 기호(쉼표, 마침표 등)을 제거, 모두 소문자로 변경
 3. 이미지 이름을 key 그 이미지에 대한 캡션 리스트를 value로 갖는 딕셔너리 만듦
'''

def clean_setence(sentence: str) -> str:
    sentence = sentence.lower() # 모든 문자를 소문자로 변환
    sentence = re.sub(r'[^a-z ]','', sentence)   # 알파벳 소문자와 공백을 제외한 문자 제거('')
    re.sub(r'\s+',' ', sentence).strip()    # 여러 개 공백을 하나로 줄이고 양끝 공백 제거
    return sentence

captions_dict = {}

for line in lines:
    line = line.strip()
    if len(line) == 0:  # 빈 줄이라면,
        continue        # 다음 줄로 넘어가기
    image_and_caption = line.split('\t')    # tab 문자 기준, 이미지 정보와 문자 분리
    if len(image_and_caption) !=2:
        continue
    # 탭으로 나눈 결과가 2가 아니라면(이미지. 캡션 하나씩 있어야 함)
    # 즉, 형식이 이상하면 스킵
    image_id_raw, caption_raw = image_and_caption   # (이미지 번호, 캡션 문장)
    image_filename = image_id_raw.split('#')[0]        # 파일이름#번호

    cleaned = clean_setence(caption_raw)

    if len(cleaned.split()) < 3:    # 단어 수가 너무 적은 문장(도움 안됨) = > 제거
        continue

    captions_dict.setdefault(image_filename, []).append(cleaned)

# 캡션이 있는 이미지의 개수
print('이미지 개수(캡션 포함):', len(captions_dict))

# 한 이미지에 어떤 캡션 들어 있는지 예시로 하나만 출력
# captions_dict.keys()  # 파일이름

sample_key = next(iter(captions_dict.keys()))
print(sample_key)

for c in captions_dict[sample_key]:
    print('-', c)

# 8000장 => 200장 사용
all_images_filename = list(captions_dict.keys())
len(all_images_filename)

subset_size = 200
if len(all_images_filename) < subset_size:
    subset_size = len(all_images_filename)

# 랜덤 샘플 뽑기
small_image_filename = random.sample(all_images_filename, subset_size)

'''
단어 사전(vocabulary) 만들기
이미지 캡셔닝에서 문장을 다루려면, 단어->숫자로 바꾸는 과정 필요
 * 특별 토큰
  - <pad> : 빈 자리를 채울 때 사용하는 토큰(길이를 맞추기 위함)
  - <start> : 문장이 시작됨을 알리는 토큰
  - <end> : 문장이 끝났음을 알리는 토큰
  - <unk> : 사전에 없는 단어를 대신하는 토큰
단어 빈도가 너무 낮은 단어는 모두 <unk>로 처리해, 사전의 크기를 적당히 줄임 
'''

# 단어 사전 구성
special_tokens = '<pad>', '<start>', '<end>', '<unk>'

word_counter = Counter()    # 각 단어가 몇 번 등장했는지 세려고

for img in small_image_filename:
    for cap in captions_dict[img]:  # 각 이미지에 대해 caption들을 순회
        for w in cap.split():       # caption에 있는 (문장의) 단어 공백기준으로 나눔
            word_counter[w] += 1    # 해당 단어 등장 빈도 1 증가

min_freq = 3    # 단어가 최소한 몇 번(3번)이상 나와야 사전에 포함시킴(threshold, 임계값)

vocab_words = [w for w, c in word_counter.items() if c >= min_freq]
print(len(vocab_words))

idx2word = []                           # index에서 단어로 바꾸어 주는 리스트
idx2word.extend(special_tokens)         # 앞쪽에 특수 토큰들을 순서대로 추가
idx2word.extend(sorted(vocab_words))    # 나머지 단어들을 정렬하여 뒤에 붙이기

word2idx = {w: i for i, w in enumerate(idx2word)}

pad_idx = word2idx['<pad>']
start_idx = word2idx['<start>']
end_idx = word2idx['<end>']
unk_idx = word2idx['<unk>']

print(len(idx2word))

vocab_words = len(idx2word)
print(vocab_words)

def sentence_to_indices(sentence: str, max_len: int=20):
    tokens = sentence.split()
    indices = [start_idx]   # 문장 시작을 의미하는 토큰 인덱스 맨 앞에 추가
    for w in tokens:
        idx = word2idx.get(w, unk_idx)  # 단어가 사전에 있으면 그 인덱스, 없으면 unk index 가져옴
        indices.append(idx)
        if len(indices) >= max_len -1:
            break
    indices.append(end_idx) # 문장 끝을 의미하는 토큰 인덱스 마지막에 추가
    if len(indices) < max_len:
        indices.extned([pad_idx]*(max_len-len(indices)))    # 남은 부분 모두 <pad> 채워
    return indices

example_caption = captions_dict[small_image_filename[0]][0]
print(example_caption)
example_indices = sentence_to_indices(example_caption, max_len=10)  # 최대 길이 10으로 제한
print(example_indices)

print([idx2word[i] for i in example_indices])