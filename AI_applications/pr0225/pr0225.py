import torch, transformers, datasets, evaluate
import numpy as np
from transformers import TrainingArguments, Trainer, DefaultDataCollator
import evaluate
import numpy as np
import torch

print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('Transformers:', transformers.__version__, "| Datasets:", datasets.__version__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

from datasets import load_dataset
beans=load_dataset('beans')
print(beans)
print(beans['train'].features)  # beans의['train'] 특징 추출
key = 'labels' if 'labels' in beans['train'].features else 'label'  # 라벨 값이 없을 경우 'label' 반환
print(beans['train'].features[key]) # beans의 ['train']의 key(labels)를 추출

from transformers import AutoImageProcessor, ViTForImageClassification
import torch

# 모델 및 프로세서 설정
MODEL = 'google/vit-base-patch16-224'   # 모델 불러오기
processor = AutoImageProcessor.from_pretrained(MODEL, use_faster=True)   

# 라벨 매핑 설정
key = 'labels' if 'labels' in beans['train'].features else 'label'
names = beans['train'].features[key].names
id2label = {i:n for i, n in enumerate(names)}
label2id = {n:i for i, n in enumerate(names)}

# 모델 로드
model = ViTForImageClassification.from_pretrained(
    MODEL,
    num_labels=len(names),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
    # ignore_mismatched_sizes => 크기가 다른 헤드(layer) 무시 => 재초기화
).to(device)

import os

num_proc = os.cpu_count()
print(num_proc)

def transform(ex):
    # 입력이 PIL 이미지 리스트 (batched=True)
    # processor가 알아서 resize, 정규화 수행, tensor 변환
    inputs = processor(images=ex['image'], return_tensors='pt')
    # 이미지 리스트를 프로세서에 전달
    # processor: 전처리 규칙

    # 결과 저장
    ex['pixel_values'] = inputs['pixel_values']
    # pixel_values: 4차원 텐서로 출력 (b, c, h, w)
    # (입력된 이미지의) 픽셀 값만 추출 => 배치에 추가
    return ex

'''
windows환경에서 생기는 버그
ValueError: You have to specify pixel_values
모델에 들어갈 이미지 데이터(pixel_value)가 없음
=> 원인 : 윈도우 멀티프로세싱의 함정

윈도우 환경에서 코어 8개(자식 프로세스)가 생성되면 일어나는 일
1. 자식 프로세스들이 코드를 읽어 내려옴
2. if __name__ == '__main__' 블록을 만남. 메인프로스세가 아니면 건너띔 
   즉, 전처리 과정을 패스 => pixel_value 생성 안됨
3. 아래에 있는 beans.set_format('torch')부터 trainer.train()까지는 if문 밖에 있으니까 
   자식 프로세스들이 각자 실행
4. 전처리가 안된 텅빈 데이터(라벨만 남아있는 데이터)를 가지고 프로세스들이 훈련을 시도하니까
   모델이 이미지(pixel_value)가 없음을 알리기 위해 계속 에러를 뱉어냄

해결 방법 : 실행코드를 모두 if __name__ == '__main__: 블록 안으로 넣어 주면 된다.
'''
if __name__ == '__main__':
    num_proc = os.cpu_count()
    print(f'사용 코어 수: {num_proc}')
    
    # 먼저 map 전처리 수행 (병렬 처리 추가)
    # remove_columns 사용, 원본 'image' 컬럼 제거 (메모리 절약)
    beans = beans.map(
        transform,
        batched=True,
        remove_columns=['image'],   # 학습에 필요 없는 원본 이미지 컬럼 삭제
        num_proc=num_proc           # 핵심: 멀티 프로세싱(속도 향상)
    )

# format 설정
# Huffing Face Trainer 쓰면 이 부분 생략 => 대신, DataCollacter가 처리하게 함
    beans.set_format('torch')

    def keep(split):
        cols = ['pixel_values', key]
        # pixel_vales 이미지 데이터, key: 라벨
        # 즉, cols는 이미지데이터, 라벨을 남김
        # split: train, val. test
        return beans[split].remove_columns([c for c in beans[split].column_names if c not in cols])
        # 지울 컬럼리스트 = 전체 컬럼 - 남길 컬럼

    train, val, test = keep('train'), keep('validation'), keep('test')

    # 정확도 지표 로드
    acc = evaluate.load('accuracy')

    # 평가 계산 함수 정의
    def metrics(p):
        predictions, labels = p
        pred = np.argmax(predictions, axis=1)
        return {'accuracy': acc.compute(predictions=pred, references=labels)['accuracy']}

    # TrainingArguments 설정
    args = TrainingArguments(
        output_dir='/content/vit_beans',
        eval_strategy='epoch',
        save_strategy='epoch',
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        learning_rate=2e-5,
        report_to='none',
        remove_unused_columns=False     # 이미지 데이터셋 컬럼 유지 위해 권장
    )

    # Trainer 초기화
    # tokenizer=processor => 이것 대신 data_collater 명시하는 게 정석
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=DefaultDataCollator(),    # 텐서 배치를 위한 Collator
        compute_metrics=metrics
    )

    # DefaultDataCollator()
    # 각 개별 샘플들을 하나씩 깔끔하게 배치(batch) tensor로 묶어주는 역할

    # 학습 시작
    trainer.train()

    print(trainer.evaluate(test))

    for i in [0,1]:
        ex = beans['test'][i]
        
        # 입력 데이터 처리
        input_tensor = ex['pixel_values'].clone().detach()
        inputs = input_tensor.unsqueeze(0).to(model.device)

        # 모델 예측
        with torch.no_grad():
            logits = model(inputs).logits
            pred = logits.argmax(-1).item()     # 정수 변환

        # 정답 라벨 가져오기
        label_key = 'labels' if 'labels' in ex else 'label'

        true_label_id = ex[label_key].item()

        # 결과 출력
        print(f'[{i} 예측: {model.config.id2label[pred]} | {model.config.id2label[true_label_id]}]')