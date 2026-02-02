'''
opencv-python 특이점
1.opencv 'pixel'값을 numpy로 표현
2.(colab) matplotlib에 이미지 표현을 맡김 
'''

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

path = r'D:\rokey\AI_applications\data\wafer.jpg'

print('-----색변환-----')
src = cv2.imread(path)
print(src.shape)                # (420, 420, 3)
cv2.imshow('Original', src)     # BGR 형식으로 표현 
cv2.waitKey(0)
cv2.destroyAllWindows()

img_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
cv2.imshow('BGR2GRAY', img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(f'차원(channel): {img_gray.ndim}')        # 차원: 3
print(f'형태(shape): {img_gray.shape}')         # 형태: (420, 420, 3)
print(f'데이터타입(data): {img_gray.dtype}')    # 데이터타입 : uint8(unsigned integer)=> pixel을 얼마나 자세히 표현해주는가     
# BGR => RGB(red, green, blue)
# opencv 기본 세팃 BGR순서 => RGB로 변환 필요

img_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
cv2.imshow('BGR2RGB', img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
# BGR => RGB 변환 함수

save_path = r'D:\rokey\AI_applications\data\wafer_complet.jpg'
cv2.imwrite(save_path, img_gray)
'''
차원별 의미
 1. 1차원(높이): 이미지의 세로(위<->아래)
 2. 2차원(너비): 이미지의 가로(좌<->우)
 3. 3차원(채널): 색상 정보 (R,G,B 3가지)

흑백 vs 컬러
 흑백 사진 = 2차원 [높이][너비]
 컬러 사진 = 3차원 [높이][너비][RGB 3개] => numpy [H, W, C] => pytorch [C, H, W]

uint8의 의미
 - unsigned: 부호(+,-)없음 (음수 없음, 0 이상만)
 - 8: 2^8=256(즉, 256가지 값으로 표현)
 - 0: 완전 어두움(검정), 255: 완전 밝음(흰색)
'''

# 차원 확인
print(src.ndim)
if src.ndim == 2:
    print('흑백사진')
elif src.ndim == 3:
    print('칼러 사진')

print('-----크기조절-----')
resized_img = cv2.resize(src, (244,244))
cv2.imshow('resized', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

resized32_img = cv2.resize(src, (32,32))
cv2.imshow('resized32', resized32_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 크기 조정시 2의 제곱수로 조정하는 것을 권장

'''
실무 TIP
이미지 데이터 증강 및 변환 등 전처리 권장
1. 이미지 로드
2. 색상 공간 변환 (데이터 성격에 맞게 RGB, GRAY 변환)
3. 이미지 크기 조절 (모델 입력크기 조정 2의 승수로 조정하는 것을 권장)
4. 데이터 타입 변환 (uint8 (0-255) => float32/64 => 정규화(255로 나누니깐 0-1))
차원 변경 (학습을 위해) : hwc => chw
'''

print(src[:2])  # (420,420)

height, width = src.shape[:2]
print(f'이미지크기: {height} * {width}')

# 사진을 224*224 크기로 만들기
if width != 224 or height !=224:
    src = cv2.resize(src, (224, 224))

'''
그림 연산
 - 사칙연산(+,-,*,/)
 - 논리연산(AND, OR, XOR, NOT)
'''
src_add = cv2.add(src, 100) # 빛의 밝기(명도 증가)
# 각 픽셀에 100이라는 숫자를 더함 => 밝아짐

src_subtract = cv2.subtract(src, 100)
# 각 픽셀에 100이라는 숫자를 뺌 => 어두어짐
fig, axes = plt.subplots(1, 2, figsize=(6,3))

axes[0].imshow(cv2.cvtColor(src_add, cv2.COLOR_BGR2RGB))
axes[0].set_title('src_add')
axes[1].imshow(cv2.cvtColor(src_subtract, cv2.COLOR_BGR2RGB))
axes[1].set_title('src_subtract')
plt.show()


s1 = 2
s2 = 0.5

src_multiply2 = cv2.multiply(src, s1)
src_multiply_half = cv2.multiply(src, s2)

# s > 1 : 이미지가 더 밝아지고, 대비(contrast) 강하게 함 => 이미지가 뚜렷해짐
# s < 1 : 이미지가 더 어두워지고, 대비가 약해짐

src_divide2 = cv2.divide(src, s1)
src_divide_half = cv2.divide(src, s2)

fig, axes = plt.subplots(2,2, figsize=(6,6))

axes[0][0].imshow(cv2.cvtColor(src_multiply2, cv2.COLOR_BGR2RGB))
axes[0][0].set_title('src_multiply2')
axes[0][1].imshow(cv2.cvtColor(src_multiply_half, cv2.COLOR_BGR2RGB))
axes[0][1].set_title('src_multiply_half')
axes[1][0].imshow(cv2.cvtColor(src_divide2, cv2.COLOR_BGR2RGB))
axes[1][0].set_title('src_divide2')
axes[1][1].imshow(cv2.cvtColor(src_divide_half, cv2.COLOR_BGR2RGB))
axes[1][1].set_title('src_divide_half')
plt.show()

# 원본 이미지 크기(src.shape)만큼 빈 공간 생성
# => 빈 공간을 채울 숫자 타입

# 노이즈를 생성 => 변환
# 노이즈에 강한 모델 생성시 활용
dst = np.empty(src.shape, src.dtype)

# for 반복문을 돌아 다니게 할거예요 (높이, 너비)
for y in range(src.shape[0]):
    for x in range(src.shape[1]):
        dst[y,x] = src[y,x] + 50    # 밝기 증가
        # 원본(src)이미지에 50을 더해서 명도(밝기) 올림

cv2.imshow('overflow', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 하지만 위의 방식은 [0,255] 범위를 넘는 경우 보정이 안되기 때문에
# 오버플로우가 발생할 수 있다. 그렇기 때문에 cv2.add를 사용하거나
# np.clip(src.astype(np.uint16)+50,0,255).astype(np.uint8)코드를 사용한다.

dst = np.clip(src.astype(np.uint16)+50,0,255).astype(np.uint8)

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(src)
plt.subplot(1,2,2)
plt.imshow(dst)

'''
논리 연산
 - AND(교집합)
 - OR(합집합)
 - XOR(~합집합)
 - NOT(원래 값 뒤집기)
'''

# 도형 생성
img1 = np.zeros((200,200), dtype=np.uint8)
img2 = np.zeros((200,200), dtype=np.uint8)

# 도형 그리기
# cv2.rectangle(사각형)
# cv2.rectangle(소스, (시작점 x,y), 끝점(x,y), 색, 선 두께)
rec = cv2.rectangle(img1, (50,50), (150,150), (125, 125, 30), -1) # 안을 채워라
# cv2.circle(소스, (원의 중심좌표 x,y), 반지름(radius), 색, 선의 두께)
circle = cv2.circle(img2, (100, 100), 70, (40, 170, 170), -1)

# matplotlib 여러개 그래프 동시 그리기
fig, axes = plt.subplots(1, 2, figsize=(6,3))

axes[0].imshow(cv2.cvtColor(rec, cv2.COLOR_BGR2RGB))
axes[0].set_title('rectangle')
axes[1].imshow(cv2.cvtColor(circle, cv2.COLOR_BGR2RGB))
axes[1].set_title('circle')
plt.show()

# AND 연산자(교집합) 겹치는 부분만 그림에 넣는다.
bit_and = cv2.bitwise_and(rec, circle)
# OR 연산자(합집합) 둘 중 하나라도 포함되는 pixel을 표현한다.
bit_or = cv2.bitwise_or(rec, circle)
# XOR 연산자(둘 다를 경우) 모두 다를 경우 1
bit_xor = cv2.bitwise_xor(rec, circle)
# NOT 연산자
bit_not = cv2.bitwise_not(rec, circle)

fig, axes = plt.subplots(2,2, figsize=(6,6))

axes[0][0].imshow(cv2.cvtColor(bit_and, cv2.COLOR_BGR2RGB))
axes[0][0].set_title('bitwise_and')
axes[0][1].imshow(cv2.cvtColor(bit_or, cv2.COLOR_BGR2RGB))
axes[0][1].set_title('bitwise_or')
axes[1][0].imshow(cv2.cvtColor(bit_xor, cv2.COLOR_BGR2RGB))
axes[1][0].set_title('bitwise_xor')
axes[1][1].imshow(cv2.cvtColor(bit_not, cv2.COLOR_BGR2RGB))
axes[1][1].set_title('bitwise_not')
plt.show()