import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

path = r'D:\rokey\AI_applications\data\wafer.jpg'
out_path = r'D:\rokey\AI_applications\pr0207'

src = cv2.imread(path)
img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
cv2.imwrite(f'{out_path}//pr0207_wafer_src.png', src)
cv2.imwrite(f'{out_path}//pr0207_wafer_img.png', img)
cv2.imwrite(f'{out_path}//pr0207_wafer_gray.png', gray)
'''
여기서 cv2와 plt로 사진을 볼 때의 차이점
cv2는 사진을 보는 색상을 BGR로 봅니다.
plt는 사진을 보는 색상을 RGB로 봅니다. 그렇기 때문에
plt에서 원본사진을 보고 싶으면 색변환을 해줘야 합니다.
그리고 사진은 RGB 3개의 채널 값을 가지고 있기 때문에
GRAY scale로 변환해주면 채널을 없앨 수 있습니다.
'''

'''
cv2.calcHist([img], [channel], [mask]. [histSize], [ranges]) 히스토그램 계산 함수
[img]: 분석할 원본 이미지
[0]: 색상 채널 grayscale일 경우는 채널이 하나이므로 0 Color일 경우 b=0, g=1, r=2
[mask]: 이미지의 특정 영역만 분석할지의 여부
=> None : 이미지 전체 분석, mask 이미지: 마스크를 만들어서 넣으면 흰색으로 칠해진 부분만 개선
[histSize] : bin의 개수 히스토그램 막대그래프 몇개 쪼갤 것인가
[ranges] : 측정할 픽셀 값의 범위
[]처리 하는 이유 : 여러 개를 한꺼번에 처리할 수 있도록 => C++ 기반으로 만들어졌다
'''

hist = cv2.calcHist([src], [0], None, [256], [0,256])

print(hist[0])
print(hist[255])
# 보정을 한다면 명도(밝기)를 높이고 컬러사진일 경우 채도를 높이는 작업이 필요

print(hist[0,0]) # 행렬의 가장 첫번째 값

# 어느 픽셀이 제일 많이 있는가(최대값 카운트)
# 최대값 의미: histMax 기준으로 다른 모든 그래프를 그림
hist_max = np.max(hist)

'''
컴퓨터 입장 좌표
y값이 클수록 아래로 내려감
컴퓨터는 y가 아래로 늘어남 => 윈도우 창 시작점(0,0)이 왼쪽 상단
막대를 위로 올리려면 전체 높이(100)이라면 전체 높이에서 빼야 함
'''
# 1단계 : 현재 밝기 값의 픽셀 개수
x = 0       # 완전 검은색
hist[x,0]   # 10,629

'''
hist = cv2.calcHist([src], [0]. None, hist_size, hist_range)
hist는 미리 계산해 둔 히스토그램 표
x=0 완전 검은색을 미리 계산해 둔 히스토그램 표인 hist에서 x번째 칸의 값 확인
'''

# 2단계: 최대값 대비 비율(0-1 사이)
# 3단계: 퍼센트로 변환(백분율, 0-100)
per = hist[x,0] * 100/ hist_max

# 4단계 정수로 변환
per = int(per)

# 5단계: y좌표 계산 (아래에서 위)
# 왜 100에서 빼나 컴퓨터는 y축이 뒤집혀 있음
y_axis = 100-per

image_bg = np.full((100,256),255, dtype=np.uint8)

for x in range(256):
    # 시작점 pt1
    pt1 = (x,100)
    # 컴퓨터는 y축이 뒤집혀 있음 높이가 100 인경우 디지털 100= 사람이 생각하는 0

    # 끝점 pt2
    pt2 = (x, 100-int(hist[x,0]*100 / hist_max))

    # cv2.line(이미지, 시작점, 끝점, 색상)
    cv2.line(image_bg, pt1, pt2, 0)

plt.imshow(image_bg, cmap='gray')
plt.show()

'''
pt= (x, 100) 의미
 - (x, 100) = (가로 위치, 세로 위치)
 - x = 0 >> (0, 100) (밝기 0 (왼쪽 끝) 아래쪽)
 - x = 50 >> (50, 100) (밝기 50 위치에서 아래쪽)
 - x = 100 >> (100, 100) (밝기 100 위치에서 아래쪽)
일반적으로 히스토그램 (밝기, 개수) 형태임
가로축(x축) : 픽셀 밝기 값 위치
 - x = 0 (아주 어두운 픽셀)
 - x = 128 (중간 밝기)
 - x = 255 (아주 밝은 픽셀)
세로축(y축) : 픽셀의 개수
'''

hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
cv2.imwrite(f'{out_path}//pr0207_wafer_hsv.png', hsv)
# HSV(Hue, Saturation, Value / 색, 채도, 명도)
# => 색상 분리

# 파란색만 추출하기
# 파란색 범위(2개) 지정

lower_blue = np.array([90,50,50])
high_blue = np.array([130,255,255])
mask = cv2.inRange(hsv, lower_blue, high_blue)
cv2.imwrite(f'{out_path}//pr0207_wafer_mask.png', mask)

# AND 연산자
# 해당하는 흰색부분만 남기기
result = cv2.bitwise_and(src, src, mask=mask)
cv2.imwrite(f'{out_path}//pr0207_wafer_result.png', result)