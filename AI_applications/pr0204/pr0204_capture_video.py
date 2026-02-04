import numpy as np
import matplotlib.pyplot as plt
import cv2

video_path = r'D:\rokey\AI_applications\data\bird.mp4'

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

# 비디오 로드 정상 여부 확인
if not cap.isOpened():
    print("ERR : 에러 발생(비디오 파일을 찾을 수가 없습니다)")
    exit()

# 영상 속성 정보 확인(fps, width, height, frame count)
# fps (frame per second: 1초 당 프레임 수)
# 30fps : 1초에 30장 사진이 지나감(영화관 일반적으로 24fps)
# width, height : 비디오 가로/세로 크기 (픽셀수)  FHD(FULL HD) 1920 * 1080
# frame count : 비디오(동영상) 전체 프레임(사진) 개수
# 30FPS, 10초 영상 : 300 프레임(사진 300장)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))          # FHD일 경우 1920
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))        # FHD일 경우 1080
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))    # 30FPS, 10초 영상 : 사진 300장

print(f'fps: {fps}, width: {width},height: {height},frame_count: {frame_count},')

# 프레임 반복 처리
frame_display_limit = 50
frames_displayed = 0

while True:
    result, frame = cap.read()
    # result: bool T/F 반환, frame: 실제 이미지 데이터(사진 한 장)

    if not result: # 마지막 프레임이면
        break

    if frames_displayed >= frame_display_limit:
        print(f"Exceeded frame display limit of {frame_display_limit}. Stopping display.")
        break

    # 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    frames_displayed +=1

    # q(quit)을 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()