import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

image_path = r'D:\rokey\AI_applications\data\copy.png'

image = cv2.imread(image_path)
cv2.imshow('image', image)

# 이진화 : 흑백(0 ,255) 채널로 이미지 변경
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)

result, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
# 50 (threshold) 임계값(50보다 어두우면 검정, 밝으면 흰색)

# 적응형 이진화
# => 영역(region)마다 다른 임계값을 사용하겠다는 의미
# cf 일반적으로 이진화라 하면 동일한 기준 사용

binary_ad = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    4
)
'''
cv2.adaptiveThreshold(
적용할 이미지,
최대값,
옵션(적응형 판단기준) 주변 픽셀의 가중평균(가우시안)
주어진조건(임계치)에 맞으면 결과를 이렇게 해줘(흰색, 검은색)
커널 크기(블록크기, 주변 영역 크기)
상수(평균/가중평균에서 빼는 값)

 단계
 (11*11) 커널 가져옴. 주변 픽셀 값 가져옴.
 (옵션) cv2.ADAPTIVE_THRESH_GAUSSIAN_C 활용, 주변 픽셀에 가우시안 평균(가중 평균) 적용, 평균 계산
 여기서 나온 평균값에서 상수 (4)를 뺀 값을 threshold T 로 사용
 픽셀 값이 T 보다 크면 >> 흰색(255), 작으면 >> 검정색(0)
 C 상수 : 값이 크면 >> threshold 낮아짐 >> 더 밝게 잡힘
 어디에 사용하나요? 로봇 라인트레이싱, OCR, 자동차번호판 인식 (조명변화에 강함)
)
'''

ret, binary_global = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

cv2.imshow('binary_ad', binary_ad)
cv2.imshow('binary_global', binary_global)

# 엣지 검출
'''
sobel fillter
1차 미분, slope(gradient)
영상의 밝기 변화량(gradinet)계산 => 물체 윤곽선(contour), 경계(edge)추출하는 필터
밝기가 급격히 변함 =>edge(경계)네 vs 변화가 없네 => 배경이네
경계의 반향과 강도 계산
'''

img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

'''
sobel filter 적용
1, 0 => 수평엣지, 0, 1 => 수직 엣지
ksize = kernerl size
'''

gx=cv2.Sobel(binary_ad, cv2.CV_32F, 1, 0, ksize=3)
gy=cv2.Sobel(binary_ad, cv2.CV_32F, 0, 1, ksize=3)

'''
binary_ad: 적응형 이진화 거친 이미지
gx(x방향) 가로 변화 감지 => 세로선 찾기
=> dx=1 x축 방향으로 미분 / dy=0 y축 방향으로 미분 안함
gy(y방향) 세로 변화 감지 => 가로선 찾기
=> dx=0 x축 방향으로 미분 암함 / dy=1 y축 방향으로 미분
cv2.CV_32F: 32비트 실수형(실수는 음수 포함 => 정밀하게 계산)
cf.uint8(일반이미지) openCV, matplotlib가 일반이미지로 인식(0-255)
'''

# 필터를 사용하여 엣지의 강도 추출
mag = cv2.magnitude(gx, gy)

'''
피타고라스의 정리활용 => 전체 엣지 강도 계산 mag=sqrt(gx^2+gy^2)
=> 수학적으로 벡터의 크기를 구하는 방법
=> 방향 상관없이 얼마나 강한 경계(edge)를 가지고 있는지 계산
'''

mag = np.clip(mag, 0, 255).astype(np.uint8)
# np.clip(mag, 0, 255) mag 부분을 이미지로 표현 [0,255] 범위로 표현하도록 강제

# Canny 필터(적용할 이미지. 낮음 임계값, 높은 임계값)
canny = cv2.Canny(binary_ad, 30, 200)
cv2.imshow('gx', gx)
cv2.imshow('gy', gy)
cv2.imshow('mag', mag)
cv2.imshow('canny', canny)

'''
Canny filter 동작방식

1. 노이즈 제거(Gaussian Blur)
2. 그래디언트 계산(Sobel Filter)
3. 비최대 억제(Non-Maximum Suppression) : 엣지의 방향을 따라 로컬의 최댓값만 남기고 나머지는 억제
4. 이중 임계값 처리(Double Thresholding) : 강한 엣지, 약한 엣지를 파악하여 약한 엣지는 강한 엣지와 연결되었을 때만 보존
5. 강한 엣지와 연결되지 않은 약한 엣지는 제거
'''

'''
코너(모서리) 검출 알고리즘

1. 해리스코너(코너) 알고리즘
2. 시-토마스 알고리즘
'''

# 해리스 코너
# 윈도우를 모든 방향으로 움직였을 때, 픽셀 값의 변화가 가장 큰 지점을 코너로 생각함

harris = cv2.cornerHarris(np.float32(gray), blockSize=2, ksize=3, k=0.04)
'''
모서리 찾기
코너나 두 방향으로 모두 강한 밝기 변화량이 존재하는 곳(특징점 추출)
코너(선과 선이 교차하는 곳) 코너일 가능성이 얼마나 높지?
점수(score) 맵(map)
score가 높으면 코너일 가능성이 높음
'''

'''
(np.float32(gray) : gradient 계산(미분) uint8 => float32
blocksize = 2 (2*2) 주변 영역 검사
=> 코너 점(point) 계산할 때 고려하는 주변 영역 크기
ksize sobel filter 커널 사이즈(3*3) 기울기(gradient) 계산
k = 0.04 민감도 (threshold 임계치)조절 (0.04-0.06)
클수록 더 민감하게 반응 => 코너가 더 많이 검출

정규화
검출 결과를 0-255 범위로 정규화 (uint8)
정규화하는 이유
=> 해리스 검출 결과가 점수 맵(scroe map이기 때문에)
(값 범위가 매우 크거나 음수가 될 수 있기 때문에 0-255로 압축 표현)
'''

harris_norm = cv2.normalize(harris, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
'''
cv2.NORM_MINMAX 최소값 최대값을 맞춰 주는 선형 정규화
최소값 0 최대값 255로 맞춰줌
'''

corner = cv2.cvtColor(binary_ad, cv2.COLOR_BGR2RGB)

corner[harris > 0.9*harris_norm.max()] = (0,0,255)
# 조건(harris 원본 응답값의 상위 10% 이상인 값)을 만족하는 인덱스(좌표)들은 빨간색으로 표시한다.

'''
goodFeaturesToTrack
 - 해리스 코너를 개선한 알고리즘
'''

src = cv2.imread(image_path)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

'''
알고리즘 적용 : goodFeaturesToTrack
 - 실무용
 - 해리스 코너는 점수맵(score map) 사용자가 직접 임계치(threshold) 설정
 - goodFeatures는 해리스코너 점수 계산한 뒤, 점수 정렬, 임계치 필터링, 거리 필터링까지 자동화
'''

# 추적하기 좋은 특징점 찾기
pts = cv2.goodFeaturesToTrack(
    gray,
    maxCorners=50,
    qualityLevel=0.01,
    minDistance=10
)
'''
cv2.goodFeaturesToTrack(
    이미지소스,
    최대 50개 특정점
    상위 1%만
    특징점 간 최소거리 10px
)
'''

if pts is not None:
    pts = pts.astype(np.uint8)
    for i in pts:
        x, y = i.ravel()    # 3차원 배열 => 1차원으로 펼침
        cv2.circle(src, (x,y), 5, (0,0,255), -1)

gd_feature = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
cv2.imshow('harris_corner', corner)
cv2.imshow('goodFeatiresToTrack', gd_feature)

'''
허프 변환(Hough Transform)
 - 직선 검출 가능
 - 원 검출 가능
'''
np.around(1,54).astype(np.uint8)
image = cv2.imread(r'D:\rokey\AI_applications\data\water_coins.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

circles=\
cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=30,
    param1=100,
    param2=30,
    minRadius=10,
    maxRadius=50
)
'''
gray 이진화된 이미지
cv2.HOUGH_GRADIENT 기울기 사용 원을 찾겠다.
dp=1.2 해상도 scaling 비율(1.0 입력 이미지와 같은 해상도) 
       => 값이 크면 속도 빨라짐, 정확도(성능) 떨어짐
minDist=30 검출된 원(중심) 사이 최소 거리 => 원 중심 간의 거리가 30이하 => 중복
param1=100 높은 임계값(canny edge upper threshold)
param2=30  투표(voting) 누적 임계값(원이라고 판단할 기준) 
           => 원의 방정식을 이용하는 것으로 추정 => 30개 이상이면 원으로 판단 
minRadius 최소반지름
maxRadius 최대 반지름
투표(voting) 많은 점(pts)이 원을 지지하면 원으로 인정
'''
if circles is not None:
    circles = np.around(circles).astype(np.uint16)

    for (x,y,r) in circles[0,:]:
        cv2.circle(image, (x,y), r, (0,255,0), 2)
        # 원의 중심좌표 (x,y) 반지름 r은 허프변환으로 찾아낸 반지름
        cv2.circle(image, (x,y), 2, (255,0,0), 2)

cv2.imshow('HoughCircle', image)

'''
circles[0,:] 의 의미 3차원에서 첫 번째 축을 0으로 고정해서 차원 하나를 인덱싱
원 그림에서 배열의 첫번째 차원(1) 무시
circles.shape 3차원 배열 (1,N,3) => 무시하면 2차원 (N,3)
여기서 첫번째 차원(1) 항상 크기가 1 (배치가 1인것과 유사)
세번째 차원(3) 3가지 정보 (0: 원의 중심 x좌표, 1: 원의 중심의 y좌표, 2: 원의 반지름 r반지름)
검출된 모든 원(n개) 정보만 선택
'''

edges= cv2.Canny(gray, 50, 150)

'''
확률적 직선 검출
이미지 edge(경계) 에서 직선(line)들을 자동 검출하는 알고리즘
=> 실제 길이 선분 형태로 반환(실무)
=> 시작점(x1, y1)과 끝점(x2,y2) 좌표로 이루어진 배열 출력

cf.  cv2.HoughLines 무한 직선 형태
HoughLinesP : P (probability) 확률

'''
image = cv2.imread(r'D:\rokey\AI_applications\data\water_coins.jpg')
from cv2.gapi import threshold
# 흑백 변환 >> 엣지 검출 >> 직선 검출
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 캐니엣지 검출
edges = cv2.Canny(gray, 50, 150)

# 확률적 직선 검출
lines =\
cv2.HoughLinesP(
    edges,                  # 캐니엣지에서 검출된 엣지 모음
    rho = 1,                # 해상도
    theta = np.pi/180,      # 해상도(각도)
    threshold = 10,         # 직선으로 간주될 수 있는 최소값
    minLineLength = 10,     # 내가 검출하려고 하는 직선의 최소 길이
    maxLineGap = 10         # 직선으로 간주되는 간격
)

# 시각화
if lines is not None:
    for line in lines:
        x1, y1, x2, y2  = line[0]
        cv2.line(image, (x1,y1), (x2,y2), (0, 0, 255), 2)

cv2.imshow('HoughLineProbability', image)
'''
없음 >> 예제를 바꿔야 함 직선이 없음
그래서 threshold 를 낮춰 noise라도 검출
rho : 원점(0,0)에서 직선까지 거리 (1px 단위로 매우 정밀하게 계산한다)
theta : 각도 단위(radian)
=> np.pi/180 = 1도(degree)
=> 각도 resolution이 크면,직선 검출 정확도 떨어짐
threshold=100, # 직선으로 간주될 수 있는 최소값(최소 투표수 100)
=> 누적 투표수가 100 이상 되어야 선이라고 간주
=> 높이면 확실한 직선만 검출 (노이즈 적어져요), 낮추면 작은 선들도 다 검출
minLineLength=10,  # 내가 검출하려는 직선의 최소 길이(길이가 10 px 미만 >> 무시)
maxLineGap=10)     # 직선으로 간주되는 간격(직선 간격 허용)
=> 중간에 끊어져 있는 선이 있어요. 선들 간의 사이가 10이하면 연결된 하나의 선으로 간주
'''

'''
OTSU : 자동으로 최적 임계값 찾기
- grayscale image => 이진화 하는 코드
- 히스토그램(픽셀 값 분포) 자동으로 분석 => 최적의 임계값( threshold) 
  => 이미지 내부에서 흑(0), 백(255) 나눠줘요
- 문서 스캔, OCR

'''
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 임계값 적용
_, binary = cv2.threshold(gray, 0,255, cv2.THRESH_OTSU)
# '_' : threshold(안 쓴다)
# 왜? 의미없어. otsh가 자동으로 최적값을 찾아줌

'''
# 컨투어 검출
# cv2.RETR_ETTERNAL 윤곽선 찾아내는 방법
# cv2.CHAIN_APPROX_SIMPLE 윤곽선을 저장하는 방법
'''

cons = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
'''
# binary : 임계값 적용해서 나온 이진화된 이미지
# cv2.RETER_LIST : (retrieve list 목록을 검색하다.) (모든 윤곽선 검색해 봐요)
# cv2.CHAIN_APPROX_NONE 모든 점 저장
# 흰색(255) 영역의 경계선 찾아요 => 리스트 형식으로 반환
'''

con_packs = cons[0] if len(cons) == 2 else cons[1]

ctr = cv2.drawContours(image, con_packs, -1, (0,255,0), 2)

cv2.imshow('Hough', circles)
cv2.imshow('Canny edges', edges)
cv2.imshow('HoughLines', image)
cv2.imshow('Contours', ctr)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
cv2.RETR_LIST
- 윤곽선 검색 방식(contour retrieval mode)
  - RETR_LIST : 모든 윤곽선 찾아서 리스트로 반환
  - RETR_EXTERNAL : 가장 바깥 윤곽선만
  - RETR_TREE : 윤곽선 계층구조까지 (부모-자식)
  - RETR_CCOMP : 2단계 계층구조까지

- cv2.CHAIN_APPROX_NONE
  - 윤곽선의 모든 점을 전부 저장
- cv2.CHAIN_APPROX_SIMPLE
  - 꼭 필요한 점(시작점, 끝점)만 저장

'''

'''
요약
1. cv2.moments
=> momnets: 이미지에서 '중심' 어디지?, '면적' 얼마지? => 데이터 수치
2. addWeighted : 두 영상이 겹쳐져요. '잔상효과' 레이어 쌓기 등 => 새로운 이미지 만들어요
=> dst = w1 * img1 + w2 * img2 + gamma
배경 위에 반투명, 투명한 레이어(UI) 올리거나 두 영상을 특징 결합시 사용
3. cv2.magnitude (벡터의 크기)
=> 엣지(경계) 검출, optical flow(광학 흐름)에 자주 사용
=> 변화의 세기 mag = sqrt(gx^2 +gy^2)
각 pixel 위치 확인 => x방향 변화량 + y방향 변화량 합쳐서 '얼마나 강한 경계선(엣지) 인가?'
---------------------------------------------------------------------------------
binary_ad
전처리 과정
1) 이미지 불러와서 이진화(흑백 변환)
2) 적응형 임계값(threshold) 처리
여기까지 전처리 한 이미지(입력값)로 canny 엣지 검출
-----------------------------------------------------------------------------------
ksize와 blocksize 차이
ksize : 필터, 마스크의 크기 (반드시 홀수만 가능: 중심점 계산해야 하기 때문 좌우/상하 차이구해야함) 3*3, 5*5 등
blocksize : adaptiveThreshold 같이 적응형 기반 계산 하기 위함 (주변 영역의 이웃의 크기)
즉 평균 구할 영역의 크기 (홀수만 가능)
------------------------------------------------------------------------------------
minDist=30,
검출된 원(중심) 사이 최소거리
=> 원 중심간 간격이 30 이하 >> 중복 아니다
=> 목적: 과잉 검출 방지
30 pixel 거리 이내 다른 원의 중심을 허용 하지 않겠다.
즉, 하나의 원에 대한 하나의 검출 결과만 보장한다
값이 작으면 서로 가까운 원들을 개별적인 원으로 검출 >> 비슷한 원 여러개가 잡힘
값이 크면 원이 너무 멀리 떨어진 경우에만 검출

param2 = 30,         
투표(voting) 누적 임계값(원의로 판단할 기준)
값이 높으면 (누가봐도) 원이야 하는 확실한 원만 검출
값이 낮으면 (아닌거 같은데) 약한 원도 검출
'''