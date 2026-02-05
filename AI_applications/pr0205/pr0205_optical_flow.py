'''
Optical Flow
 - 연속된 두 프레임 사이에서 픽셀이 어디로 움직였는지 계산
 - 물체 속도 방향
 - 자율주행, 드론, 동작인식
'''
import os
import numpy as np
import cv2
import time
out_dir = r'D:\rokey\AI_applications\pr0205'

# 1. 초기설정 및 변수 정의
video_path = r'D:\rokey\AI_applications\data\newyork.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("오류: 비디오 파일을 열 수 없습니다. 파일을 업로드했는지 확인하세요.")
    exit()

delay = int(1000/30)
# real-time 실시간 영상에서 부드러운 재생 원하기 때문에 필요한 타이밍

'''
추적 경로를 그리기 위한 랜덤 색상(200개 코너점 대응하는 색상)
np.random.randint(0, 255, (200, 3))
0-255 구간 3channel(BGR) 색상 200개 생성
'''

color = np.random.randint(0, 255, (200, 3))

lines = None    # 추적선(이동경로) 그릴 이미지 저장 변수(초기화: 첫 프레임에서 진행)
prevImg = None  # previous image 이전 프레임 저장 변수(grayscale image)

termcriteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
'''
calOpticalFlowPryLK() 중지 요건 설정(Termination Criteria)
(cv2.THRM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 최대 반복 횟수(10), 오차 임계값(0.03))
cv2.TERM_CRITERIA_EPS: 변화량(오차)가 epsilon보다 작으면 종료 
=> 움직임이 0.03 px 미만 => 수렴 (error 오차가 충분히 작아지면 멈추는 조건)
cv2.TERM_CRITERIA_COUNT: 반복 횟수가 maxCount에 도달하면 종료
=> 최대 반복(iteration) 횟수(10)
변화량이 0.03보다 작아지거나 반복횟수가 10보다 많아지면 종료
'''

# 프레임 처리 개수 제한
frame_count = 0
MAX_FRAMES_TO_PROCESS = 150
DISPLAY_EVERY_N_FRAMES = 20

print(f"광학 흐름 추적 시작 (최대 {MAX_FRAMES_TO_PROCESS} 프레임, {DISPLAY_EVERY_N_FRAMES} 프레임마다 출력)...")

'''
예제
p1, st, err = cv2.calcOpticalFlowPyrLK(
    old_gray,               이전 사진 
    gray,                   현재 사진
    p0,                     position 추적하고 싶은 점들의 좌표
    winSize=(15,15),        (15,15) 추적한 윈도우 사이즈 (주변 영역 15*15 영역 비교 이동방향 개선)
    maxLevel=2,             이미지 피라미드 레벨 (영상 축소 단계 수)
    criteria=termcriteria   우리가 설정한 정지조건(기준) (최대 10번 또는 오차 0.03 미만)
)
cv2.calcOpticalFlowPyrLK(루카스-카나데 광학 흐름) 물체 이동 위치 추적
p1 : 추적된 새로운 좌표값
st = 1 성공 0 실패
err = 0.01 오차가 거의 없음
'''

# 비디오 처리
while cap.isOpened() and frame_count < MAX_FRAMES_TO_PROCESS:
    ret, frame = cap.read()

    if not ret:
        break

    # 현재 프레임 (사진) 복사 => 추적결과 그릴 이미지 준비
    img_draw = frame.copy()
    # optical flow 계산 위해 현재 프레임(사진)을 grayscale로 변환 => 채널 삭제
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 최조 프레임 처리(추적 시작)
    if prevImg is None:
        prevImg = gray  # 현재 gray 이미지를 '이전 이미지'로 저장
        # 추적선을 그릴 검은색 배경 이미지 생성 (원본 프레임과 동일 크기)
        lines = np.zeros_like(frame)    

        prevPt = cv2.goodFeaturesToTrack(prevImg, 200, 0.01, 10)
        # Shi_Toamsi 알고리즘으로 추적 시작할 코너점 200개 검출
        # (이전 이미지, 최대 코너점 수, 품질 임계값(0.01: 최고 코너 1%), 코너 간 최소거리 10px)
        # prevPt :  코너 점의 목록
    
    # 두번째 프레임 이후 처리(추적 진행중)
    else:
        nextImg = gray   # 현재 gray 이미지를 '다음 이미지'로 저장

        # 루카스카나데 optical flow
        nextPt, status, err = cv2.calcOpticalFlowPyrLK(prevImg, nextImg,
                                                       prevPt, None, criteria=termcriteria)


        # 추적에 성공한 코너점(status == 1) 선별
        prevMv = prevPt[status == 1]    # 이전 프레임에서 추적 성공한 점
        nextMv = nextPt[status == 1]    # 현제 프레임에 대응하는 점 

        # 추적 성공한 모든 쌍에 대해서 반복
        for i, (p, n) in enumerate(zip(prevMv, nextMv)):
            # 코너점 좌표 추출(배열 구조 해제)
            px, py = p.ravel()
            nx, ny = n.ravel()

            cv2.line(lines, (int(px), int(py)), (int(nx), int(ny)), 
                     color[i % len(color)].tolist(), 2)
            # 시작점: 이전 위치(px, py), 끝점 (nx, ny)
            # color[i % len(color)].tolist() : 코너점 i 에서 할당된 랜덤 색
            # tolist() : 순수 python 리스트 배열로 바꿔 준다.
            # 이전 좌표에서 다음 좌표로 선을 그리고 색은 color에서 골라서 써라

            cv2.circle(img_draw, (int(nx), int(ny)), 2, color[i % len(color)].tolist(), -1)

        # 누적된 추적선이 그려진 lines 이미지와 현재 프레임(img_draw)을 합성
        # => 추적 경로가 비디오 프레임 위에 나타남
        img_draw = cv2.add(img_draw, lines)

        # 다음 루프 위해 현재 프레임과 코너점 => 이전 변수로 이동
        prevImg = nextImg

        prevPt = nextMv.reshape(-1, 1, 2)
        # prevPt를 nextMv 형태로 맞춰 줘야 함
        # (N, 2) => (N, 1, 2)
        # N: 점의 개수, 1: 형식상 묶음, 2:(x,y) 좌표
        
        if frame_count % DISPLAY_EVERY_N_FRAMES == 0:
            # cv2.imwrite(f'{out_dir}\\pr0205_opticalflow_frame_{frame_count}.png', frame)
            # cv2.imwrite(f'{out_dir}\\pr0205_opticalflow_lines_{frame_count}.png', lines)
            cv2.imwrite(f'{out_dir}\\pr0205_opticalflow_image_{frame_count}.png', img_draw)
            print(f'saved: {frame_count}')


        frame_count += 1

'''
cv2.calcOpticalFlowPyrLK
이전 프레임의 점들을 기준으로 다음 프레임에서 대응되는 점들위 위치를 추적하는 함수
        
원리
 가정
  1. 밝기(색)가 짧은 시간엔 거의 안변한다.
   - 같은 물체의 점이면 다음 프레임에서도 비슷한 밝기 패턴을 가짐
  2. 움직임이 작다 (또는 작게 쪼개서 본다.)
   - 한 점 주변 작은 윈도우 안에서 가장 잘 맞는 이동을 찾을 수 있음
  => 이미지를 작은 해상도부터(피라미드)보면서 큰 이동을 먼저 대충 잡고
     원래 해상도에서 정밀하게 다듬음
         
 사용 상황
  1. 특징점을 따라가며 움직임을 추적할 때
  2. 카메라 흔들림 보정, 모션 분석, 트래킹의 기초 단계

 한계
  1. 특징점이 가려지거나 흐려지거나 빛이 확 바뀌면 실패 가능성 높아짐
  2. '점' 추적이라서 물체가 단색이고 특징이 없으면 점을 잡기 어려움
''' 
