import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

image_path = r'D:\rokey\AI_applications\data\beau_3.png'
target_path = r'D:\rokey\AI_applications\data\new_target.png'

# 사진 불러오기
image = cv2.imread(image_path)
target_image = cv2.imread(target_path)

# 사진이 잘 나오는 지 확인하기
if image is None:
    print(f'Error: Could not load image from {image_path}')
elif target_image is None:
    print(f'Error: Could not load target_image from {target_path}')
else:
    # BGR => RGB 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    target_image_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_rgb)
    axes[0].set_title('Source Image')
    axes[1].imshow(target_image_rgb)
    axes[1].set_title('Target Image')
    plt.show()    

# 1.그레이스케일로 변환
image = cv2.imread(image_path)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

target = cv2.imread(target_path)
target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
target_resized = cv2.resize(target_gray, (200, 240))
'''
타겟의 회전, 크기 변환, 밝기 변화가 많은 영향을 미침
템플릿 크기 200*240px 조정
중요: 템플릿 매칭은 크기가 정확하게 맞아야 작동

템플릿을 소스 이미지에 부착한 뒤 움직이면서 맞는 객체를 찾는 것
즉 탬플릿 크기보다 작은 객체는 발견하지 못함
'''

# 2. 템플릿의 너비와 높이를 찾음
w, h = target_gray.shape[::-1]
'''
원래 shape는 (높이, 너비)순서임
[::-1] 역순 => (너비, 높이) 변환
나중에 사각형 그릴 때 필요하기 때문에 변환
'''

# 3. 템플릿을 이미지에서 매칭해서 찾기
result = cv2.matchTemplate(image_gray, target_gray, cv2.TM_CCOEFF_NORMED)
'''
매칭 옵션
SQDIFF(픽셀 제곱차이): 작을수록 유사
CCORR(픽셀 곱의 합): 클수록 유사
TM_CCOEEFF_NORMED(코사인 유사도 정규화): 클수록유사
NORMED(일정한 범위 내로 만듦): 정규화 (0-1 범위로 만들기)
result: 유사도 점수
'''

# 매칭 결과에서 최소값, 최대값, 최소값 위치, 최대값 위치를 찾으려고 함
min_val, max_Val, min_loc, max_loc = cv2.minMaxLoc(result)
'''
min_val: 가장 낮은 유사도 값
min_loc: 가장 낮은 유사도 값의 위치(가장 안 비슷한 위치)

max_val: 가장 높은 유사도 값
max_loc: 가장 높은 유사도 값의 위치(가장 비슷한 위치)
즉 가장 유사도 값이 높고 비슷한 위치(x,y) 좌표를 찾음
'''

# 시각화
top_left = max_loc  # 가능성이 높은 지역의 왼쪽위 모서리
# top_left = (x,y)형태라서 top_left[0] x좌표, top_left[1] y좌표
bottom_right = (top_left[0]+w, top_left[1]+h)
# 왼쪽 위 모서리를 기준으로 +너비, +높이해서 구한 오른쪽 아래 모서리 [x,y,w,h]
matched = image.copy()
cv2.rectangle(matched, top_left, bottom_right, (0,255,0), 2)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(image)
axes[1].imshow(result)
axes[2].imshow(matched)
plt.show()
'''
결과는 탬플릿을 소스 이미지 전체에 대봤을 때의 유사도 분포
x 400-500구간에 유사도가 높은 위치들이 많이 모여 있을 가능성이 높다.
'''