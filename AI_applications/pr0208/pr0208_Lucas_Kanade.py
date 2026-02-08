import cv2
import numpy as mp
import time

# 1. 초기 설정 및 비디오 읽기
video_path = r'D:\rokey\AI_applications\data\newyork.mp4'
out_dir = r'D:\rokey\AI_applications\pr0208'

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"오류: 비디오 파일 '{video_path}'을 열 수 없습니다.")
    print("영상 경로가 정확한지 확인해 주세요.")
    exit()

# 2. 첫 프레임에서 특징점 찾기
ret, old_frame = cap.read() # 첫 프레임 읽기

# 프레임을 제대로 읽었는지(ret=True) 한번 더 확인
if not ret or old_frame is None:
    print("오류: 첫 번째 프레임을 읽지 못했습니다. 비디오 파일이 손상되었을 수 있습니다.")
    cap.release()
    exit()

# 첫 프레임 흑백으로 변환
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Shi-Tomasi 코너 검출 알고리즘 사용해 특징점 찾기
corners = cv2.goodFeaturesToTrack(
    old_gray,
    maxCorners=100,     # 최대 특징점 개수
    qualityLevel=0.3,   # 특징점 품질 레벨 (0.0-1.0)
    minDistance=7       # 특징점 간 최소 거리
)
'''
Shi-Tomasi 코너 검출 알고리즘
cv2.goodFeaturesToTrack
maxCorners : 최대 100개의 가장 강한 특징점만 반환 
qualityLevel : 특징점 품질 70% 이상 검출
minDistance : 특징점 간의 최소한의 거리 이내의 다른 특징점은 지워버림(너무 붙어있지 않게 퍼트리는 역할)

Shi-Tomasi 코너 검출 알고리즘이란?
Harris cormer detection을 개선한 버전
 - 기본 원리 : 이미지에서 작은 윈도우를 움질일 때, 모든 방향으로 픽셀 변화가 큰 곳이 코너라고 판단
 - Harris와의 차이점
  * 수학적으로 코너를 판별할 때 '고유값'이라는 것을 사용
  * Harris: 두 고유값이 조합을 복잡하게 계산해서 점수를 측정
  * Shi-Tomasi: 그냥 두 고유값 중 작은 값이 특정 기준보다 크면 코너다 라고 단순화함
 - 장점: 계산이 더 단순하면서도, 추적하기 좋은 특징점을 더 잘 찾아냄

cv2.goodFeaturesToTrack
작동 순서
1. 점수 계산 : 코너 점수(강도) 계산
2. 내림차순 정렬
3. 최대 개수 제한(maxCorner) : 상위 N개만 남김
4. 품질 커트라인(qualityLevel) : 1등 코너의 %도 안되는 약한 코너들은 탈락
5. 거리두기(minDistance): 가장 강한 코너부터 시작해서, 그 주변 minDistance 안에 있는 약한 코너 지움
결과 : 화면 전체에 골고루 퍼져 있는 가장 선명하고 확실한 코너의 좌표를 반환
       그래서 광류(Optical Flow)나 물체 추적을 시작할 때 첫 단추로 가장 많이 쓰이는 함수
'''

# 3. Lucas_Kanade parameter 설정
lk_parmas = dict(
    winSize=(15,15),    # 윈도우 사이즈(크기)
    maxLevel=2,         # 피라미드 레벨
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)   # 종료조건
)

# 이미지 피라미드(coarse to fine strategy)
# 몇 단계까지 사용하는지 설정(0: 원본, 1:1단계 다운 샘플링, 2: 2단계 다운샘플링)

'''
Lucas-Kanade
물체가 아주 조금 움직였다는 가정하에 작동 그렇기 때문에 
물체가 갑자기 빠르게 움직이거나 크게 이동하면 추적에 실패
=> 이 문제를 해결하기 위해 이미지 피라미드 기술을 사용

1. 이미지 피라미드의 정의
이미지 피라미드는 하나의 원본 이미지를 다양한 해상도(크기)로 줄여서 차곡차곡 쌓아 올린 구조
 구조
  - 바닥 (Level 0): 원본 이미지(가장 크고 선명함, 해상도 높음)
  - 중간 (Level 1,2...): 점점 작아지는 이미지들
  - 꼭대기 (Level N): 아주 작음 썸네일 같은 이미지(해상도 낮음)

2. 다운 샘플링(downsampling)
이미지 피라미드의 윗 단계(작은 이미지)를 만들기 위해 사용하는 기술
샘플(픽셀)의 수를 줄여서 이미지를 축하는 과정
보통 가우시안 피라미드 방식을 가장 많이 사용함
과정
 1. 블러링(Gayssian Blur) : 이미지를 흐릿하게 만듦(갑자기 픽셀을 빼면 
   이미지가 깨져보이는 '엘리어싱' 현상 방지)
 2. 짝수 행/열 제거(Decimation) : 짝수 번째 가로줄과 세로줄을 삭제
 3. 결과: 가로, 세로 길이가 각각 절반인 이미지 생성

3 Lucas-Kanade 작동원리
큰 움직임을 속이기 위해 이미지를 작게 만듦
작동 순서:Coarse-to-Fine(거친 것에서 정밀한 것으로)
 1. 꼭대기 층(작은 이미지)에서 계산: 먼저 가장 작은 이미지에서 대략적인 움직임 계산
 2. 아래층으로 전당(Upsampling): 계산된 움직임 정보를 아래층(조금 더 큰 이미지)으로 전달
 3. 보정(Refine): 전달받은 정보를 바탕으로 더 큰 이미지에서 정밀하게 위치 수정
 4. 반복
'''

# 4. 비디오 처리 루프, 광학 흐름(optical flow) 계산

frame_count = 0
MAX_FRAMES_TO_PROCESS = 150
DISPLAY_EVERY_N_FRAMES = 20

track_frame = None

while cap.isOpened() and frame_count < MAX_FRAMES_TO_PROCESS:
    ret, frame = cap.read()

    if not ret or frame is None:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # optical flow 계산, 이전 프레임(old_gray)의 특징점(corners)이 
    # 현재 프레임에서 어디로 이동했는지 추정
    new_corners, status, error = cv2.calcOpticalFlowPyrLK(
        old_gray, frame_gray, corners, None, **lk_parmas
    )
    '''
    **lk_parmas 앞에 **의 의미
    lk_params의 딕셔너리 표장을 뜯어서 내용을 함수에 전달하는 역할을 한다.
    장점: 가독성, 재사용성(), 관리 용이
    '''

    # 좋은 점들만 선택 (status = 1 => 성공적으로 추적된 점)
    good_new = new_corners[status == 1]
    good_old = corners[status == 1]

    # 한 번만 초기화
    if track_frame is None:
        track_frame = frame.copy()  # first_frame only
    else:
        track_frame = track_frame
    
    # 움직임 그리기
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel().astype(int)  # 현재 위치
        c, d = old.ravel().astype(int)  # 이전 위치
        # 픽셀 좌표는 정수여야 하니깐 dtype을 int로 변환

        # 이전 위치와 현재 위치를 이어주는 초록샌 선 그리기
        cv2.line(track_frame, (a,b), (c,d), (0,255,0), 2)

        # 현재 위치에 빨간색 점(원) 그리기
        cv2.circle(track_frame, (a,b), 5, (0,0,255), -1)
    
    if frame_count % DISPLAY_EVERY_N_FRAMES == 0:
        cv2.imwrite(f'{out_dir}\\pr0208_Lucas_Kanade_{frame_count}.png', track_frame)
        time.sleep(1)

    # 업데이트, 다음 루프를 위해 현재 프레임을 이전 프레임으로 설정
    old_gray = frame_gray.copy()
    corners = good_new.reshape(-1, 1, 2)    # 자동 계산

    '''
    corners = good_new.reshape(-1, 1, 2)  (N,1,2)
    good_new : 성공적으로 추적된 좌표 (N,2)
    개수, 채널, 좌표
    -1 : 추적에 성공한 특징점 전체 개수 (작동계산)
    1 : 각 점(하나의 채널로 간주)
    2 : 각 특징점의 x,y좌표
    '''

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

'''
위의 코드는
Shi-Tomasi 알고리즘을 통해서 코너를 검출합니다.
그리고 Lucas-Kanade 알고리즘을 사용하여 특징점(코너)의 이동을
추적하여 선으로 이어줍니다.
그래서 결과에 나온 사진에 나온 점은 특징점을 의미하고
그 점들을 이은 선들은 특징점의 이동을 나타내는 겁니다.
'''