import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

image_path = r'D:\rokey\AI_applications\data\puppy.jpg'
out_dir = r'D:\rokey\AI_applications\pr0207'

src = cv2.imread(image_path)
image = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

# blur(정용할 이미지, 커널 크기)
# blur() : 모든 픽셀에 대한 가중치 부여
# 커널 : 필터
dst = cv2.blur(src, (11,11))
cv2.imwrite(f'{out_dir}\\pr0207_blur_kernel_sized11.png', dst)

'''
커널크기에 따라 blur 효과 달라짐
3*3 : 약간 흐림
5*5 : 중간 흐림
11*11 : 많이 흐림

작은 커널(3*3) : 주변 9개의 픽셀의 평균
 - 가까운 픽셀만 영향 받음
큰 커널(11*11) : 주변에 121개 픽셀의 평균
 - 멀리 있는 픽셀도 영향 받음

커널 크기 규칙
 - 홀수만 사용
 - 실제 사용 (3*3), (5*5), (7*7), (11*11)
 - 홀수인 이유
   - 중앙(값)이 명확 => 명확한 중심점이 존재 
'''
# 원본 이미지: BGR 이미지
cv2.imwrite(f'{out_dir}\\pr0207_image.png', image)
# 소스 이미지: BGR => RGB뱐환
cv2.imwrite(f'{out_dir}\\pr0207_src.png', src)
'''
GaussianBlur(image, ksize, sigmaX)
sigmaX : 표준 편차=> openCV가 자동 계산 블러링이 얼마나 진하게 또는 넓게 적용될지 제어
작을 때는 가까운 이웃 픽셀만 영향(블러링 효과 약함) 
클 때 멀리 있는 픽셀까지 영향을 줌(블러링 효과 강함)
중앙 픽셀에 더 큰 가중치 부여 => 평균에 몰려있기 때문
이미지를 전체적으로 부드럽게 만들거나, 일반적으로 노이즈를 줄일 때 사용
'''
gauss = cv2.GaussianBlur(src, ksize=(11,11), sigmaX=0)
cv2.imwrite(f'{out_dir}\\pr0207_GaussianBlur.png', gauss)

'''
medianBlur(image , kernel_size)
median(중위수, 중앙값) 사용: 평균의 큰 단점인 이상치
이미지에서 이상치란 noise
색이 튀어있는 노이즈를 제거할 때 성능이 뛰어남 ex) 소금, 후추
edge(경계선) 보존능력 윤곽선을 꽤 선명하게 유지, 잡티는 없에고 싶지만 테두리는 살리고 싶을 때 사용
새로운 값을 만들어 내지 않음
gaussian : 10과 20의 평균 15라는 새로운 값을 만들어냄
median : 10,10,10,20,20 중 하나인 10을 사용 원본 이미지에 존재하는 픽셀값만 사용해 데이터 손상이 적음
'''
median = cv2.medianBlur(src, 3)
cv2.imwrite(f'{out_dir}\\pr0207_medianBlu.png', median)

salt_path = r'D:\rokey\AI_applications\data\salt.jpg'

salt = cv2.imread(salt_path)
'''
1. 이진화(binary) 0, 1 => 흑백 / 임계값(기준값) 넘으면 255(흰색), 모자라면 0(검은색)
cv2.threshold(image, 임계값, 최대값, 스타일)
cv2.THRESH_BINARY 이진화 규칙 지정
'''

result, binary_image = cv2.threshold(salt, 200, 255, cv2.THRESH_BINARY)
cv2.imwrite(f'{out_dir}\\pr0207_threshold_binary.png', 
            binary_image)

'''
팽창과 침식 적용
 - erosion 침식
  * 흰색 영역(255)의 외각을 깎아내는 연산
  * 커널이 완전히 흰색을 포함하는 영역만 유지, 나머지는 검정(0)으로 바꿈
  * 객체가 작아짐(외곽이 깎여서), 작은 노이즈 제거(작은 점 형태 흰색 잡음 제거)
 - dilation 확대(팽창)
  * 흰색 영역(255) 넓혀줌  
  * 커널이 1개 라도 흰색 만나면 중심 픽셀을 흰색으로 확장
  * 객차가 커짐(빈 공간 채워줌), 끊긴 선 연결(문자, 윤곽선 연결), 구멍 채움

실무 TIP
1. Binary(0-255)
2. erosion(줄어들고 잡음 제거) or deiltion(커짐 끊김 영역 연걸)
 - 일반적으로 외부 노이즈가 문제면 closing 적용
 - 내부 구멍이 문제면 closing 적용
 - 둘 다 문제면 opening 사용 후, closing
 - opening : erosion => dilation(점 noise 빼고, 구멍 채워 줌)
  * noise 제거한 후 모양 유지
 - closing : dilation => erosion
  * 끊긴 윤곽선 연결 구멍 채움 
'''

# 커널 만들기
# np.empty(사이즈), np.zeros(사이즈)
# 커널 사이즈 만큼 비어있는 객체 생성 / 사이즈 크기 만큼 0으로 채워진 객체 생성

kernel = np.ones((3,3), np.uint8)

'''
cv2.erode(img, kernel_size, iterations)
cv2.dilate(img, kernel_size, iterations)
iterations = 몇번 수행할까 => 효과의 반복 횟수(강도)
'''
erode_image = cv2.erode(binary_image, kernel, iterations=1)
dilate_image = cv2.dilate(binary_image, kernel, iterations=1)
cv2.imwrite(f'{out_dir}\\pr0207_erode.png', erode_image)
cv2.imwrite(f'{out_dir}\\pr0207_dilate.png', dilate_image)