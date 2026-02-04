import cv2
import numpy as np

video_path = r'D:\rokey\AI_applications\data\greenball.mp4'

def track(image):
    # BGR 이미지(numpy array) 입력 => 초록색 물체 중심좌표(centroid) 추적

    # 1) 노이즈 제거 => blur처리 (가우시안 블러)
    blur = cv2.GaussianBlur(image, (5,5), 0)
    # (5, 5) 5*5 filter kernel size, 0 : 표준 편차 자동계산

    # 2) BGR => HSV 색 공간 변환
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    '''
    HSV(Hue, Saturation, Value)
    hue: 색상(빨강, 초록, 파랑...) / saturation : 채도 / value(명도)
    HSV 조명이 바뀌어도 색상 찾기가 쉬움
    BGR 컴퓨터 중심(샘 검출, 파랑, 초록, 빨강 위주, 조명 밝기 영향을 많이 받음) 
    => 추적할 때는 HSV로 바꿈
    '''
    # 3) 초록색 범위 설정(필요하면 값 조정 가능)
    # 연한 녹색 - 진한 녹색 범위 설정
    lower_green = np.array([40, 70, 70])    # HSV 색상 40, 채도 70, 명도, 70
    upper_green = np.array([80, 200, 200])  # HSV 색상 80, 채도 200, 명도, 200

    # 4) 마스크(mask) 생성
    mask = cv2.inRange(hsv, lower_green, upper_green)
    '''
    초록색 부분만 하얀색 (255), 나머지는 검은색(0) 만들기
    초록색 부분만 오려내는 효과 발생

    lower green - pixel value - upper green
    inRange: 이 범위 안에 있으면 => 255(흰색) / 없으면 0(검정색)
    '''

    # 5) 마스크 다시 블러링(잡음 noise 제거)
    bmask = cv2.GaussianBlur(mask, (5,5), 0)
    # bmask (binary mask) 이미지가 0(code 0) 또는 255(code 1) 구성된 영상 (흑백영상)

    # 6) 모멘트(moments) 이용하여 중심좌표 계산
    moments = cv2.moments(bmask)
    '''
    key-value {}로 저장 (m00: ... 면적(area))
    m00 = 전체 면적(하얀 픽셀 개수)
    '''
    m00 = moments['m00']

    # 중심좌표(centroid) 초기화
    centroid_x, centroid_y = None, None

    # 만약 면적이 0이 아니라면(면적이 존재한다면)
    if m00 !=0:
        centroid_x = int(moments('m10')/m00)    # 한 영역의 중심 x좌표
        centroid_y = int(moments('m01')/m00)    # 한 영역의 중심 x좌표
        '''
        m10, m01 의미(1차 모멘트)
        m10 : x축 방향에 대한 pixel 강도의 합
        m01 : y출 방향에 대한 pixel 강도의 합
        중심좌표(centroid) 공식
        cx(객체의 가로 중심 좌표) = m10 / m00(전체 면적)
        cy(객체의 세로 중심 좌표) = m01 / m00(전체 면적)        
        '''

    '''
    기본값: 중심 없음
    => 초록색 못 찾으면
    -1을 쓰는 이유 유효하지 않은 좌표(x좌표, y좌표 >=0)
    즉, 이미지 화면에 보이는 것 (0,0)~(w,h)
    center pint 없으면 객체 찾지 못함
    객체가 있다면(중심점 있다면) => (cx, cy)        
    '''
    crt = (-1,-1)

    # 중심이 계산된 경우
    if centroid_x is not None and centroid_y is not None:
        crt = (centroid_x, centroid_y)
        # 이미지 위에 중심 점 표시(검은색 점)
        cv2.circle(image, crt, 4, (0,0,0), -1)
    
    cv2.imshow(image)

    return crt

# 메인 실행 부분 (동셩상 캡처)
if __name__ == '__main__':
    # 프로그램 시작점(이 파일을 직접 실행할 때만 동작)
    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        print('동영상을 열 수 없습니다.')
    else:
        frame_idx = 0

        while True:
            okay, image = capture.read()

            # 비디오가 마지막이면
            if not okay:
                print('영상 끝까지 재생 완료')
                break

            # 초록색 공 추적
            ctr = track(image)
            print(f'Frame {frame_idx}: centroid = {ctr}')

            frame_idx += 1

        capture.release()

'''
결과를 보면 점이 공을 트래킹 하는 것이 아니라
뒤 배경 코트를 트래킹하는 것을 확인 할 수 있다.
이는 색의 범위를 잘 못 정의해서 생기는 현상으로 해석할 수 있을 듯 하다.
'''