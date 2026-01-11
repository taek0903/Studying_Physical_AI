# xor.py
# D:\rokey\python\ch19\sklearn\xor\xor.py

from sklearn import svm
from sklearn import metrics

# 1. 전처리

# 데이터 학습하기
clf =svm.SVC()      # 2.알고리즘 선택

# 3. 학습/훈련 => 모델 생성
# fit(학습데이터, 학습레이블)
clf.fit([
        [0,0],
        [0,1],
        [1,0],
        [1,1]
    ],
    [0,1,1,0]
)

# 4. 데이터 예측하기
# predict(테스트데이터)
pre=clf.predict([
        [0, 1],      # 1
        [1, 1]       # 0
    ])      # 예측 => 답
print(pre)

# 5. 정확도 평가
ac_score = metrics.accuracy_score([1,0], pre)
print(f'정답률: {ac_score}')