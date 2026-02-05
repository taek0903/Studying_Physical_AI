import numpy as np
import cv2
import time

video_path = r'D:\rokey\AI_applications\data\newyork.mp4'
cap = cv2.VideoCapture(video_path)  # 비디오 캡처 객체 생성

# 1. 초기설정
if not cap.isOpened():
    print(f"오류: 비디오 파일 '{video_path}'을 열 수 없습니다. 파일을 업로드했는지 확인하세요.")
    exit()  # 열 수 없으면 프로그램 종류

# 2. 배경제거 객체 생성
# MOG2(Mixture of Gaussians, 개선버전) 알고리즘을 사용한 배경 제거 객체 생성으로 수정
fgbg = cv2.createBackgroundSubtractorMOG2()
out_dir = r'D:\rokey\AI_applications\pr0205'
'''
MOG2(mixture of gaussian2)
각 픽셀 색상변화를 가우시안 분포(정규분포) 모델링
배경(background:bg)이 고정되어 있거나 오래 머무는 색상 => 배경학습
전경(frontground:fg) 갑자기 툭 나타난 색상 => 움직이는 물체로 판단
가우시안 혼합 모델 활용
'''

# 3. 비디오 처리 루프

# 특정 프레임 수만 처리하고 결과를 정적으로 표시
frame_count = 0
MAX_FRAMES_TO_PROCESS = 100 # 테스트를 위해 최대 처리 프레임 수를 100으로 제한
DISPLAY_EVERY_N_FRAMES = 20 # 20 프레임마다 결과 출력

print(f"비디오 처리 시작 (최대 {MAX_FRAMES_TO_PROCESS} 프레임)...")
# fps => frame per second 1초당 프레임 개수

while cap.isOpened() and frame_count < MAX_FRAMES_TO_PROCESS:
    result, frame = cap.read()
    # 동영상에서 한 프레임(사진 1장)읽기 (result: 성공여부, frame: 실제 이미지)

    if not result:
        break

    # 배경제거 마스크 계산
    fgmask = fgbg.apply(frame)
    '''
    frame에서 배경(검정 0)제외
    => 움직이는 객체 (전경만) 흰색(255)으로 표시한 마스크(흑백 마스크) 생성
    마스크 값
    0(검정) => 배경, 움직이지 않는 부분
    255(흰색) => 전경, 움직이는 개체
    127(회색) => MOG2가 가진 특별 기능(그림자)
    '''

    # 특정 간격의 프레임만 표시 => 작동 확인
    if frame_count % DISPLAY_EVERY_N_FRAMES == 0:
        cv2.imwrite(f'{out_dir}\\pr0205_debug_frame_{frame_count}.png', frame)
        cv2.imwrite(f'{out_dir}\\pr0205_debug_mask_{frame_count}.png', fgmask)
        print(f'saved: {frame_count}')

    frame_count += 1
cap.release()
cv2.destroyAllWindows()
'''
cv2.createBackgroundSubtractorMOG2의 알고리즘안에
마스크 객체가 내부적으로 정한다.
0(검정, 배경), 255(흰색, 전경 움직이는 물체), 127(회색, 그림자, 옵션)
'''