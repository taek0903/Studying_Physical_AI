'''
CamShift
- Meanshift 의 업그레이드 버전
- 객체가 크기가 바뀌거나 회전해도 추적가능
- MeanShift와 차이점 MeanShift 는 고정크기 박스이나, CamShift는 적응형 크기, 회전 가능한 박스
'''

import numpy as np
import cv2
import time
out_dir = r'D:\rokey\AI_applications\pr0208'

# 1. 변수 초기
roi_hist = None # 추적 객체 히스토그램 저장 변수
video_name = 'CamShift Traking'

# CamShift에 필요한 초기 추적 영역 좌표 전역 초기화
# Note : 이 값을 수정하여 추적할 객체의 초기 위치와 크기를 조정 가능
x,y,w,h = 50, 300, 100, 100
# 사각형 꼭지점 좌표 좌상:[50, 300], 우상:[150,300], 우하:[150, 400], 좌하[50, 400]

track_window = (x,y,w,h)

# CamShift 중지요건 (Termination Criteria)
# 오차(EPS) 또는 반복 횟수(COUNT) 중 하나라도 충족되면 중지
termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
# cv2.TERM_CRITERIA_EPS(허용오차): 1 cv2.TERM_CRITERIA_EPS(반복 횟수):10

# 2. 비디오 캡처
video_path = r'D:\rokey\AI_applications\data\top-down.mp4'
cap = cv2.VideoCapture(video_path)
delay = int(1000/24)

if not cap.isOpened():
    print("오류: 비디오 파일을 열 수 없습니다. 파일을 업로드했는지 확인하세요.")
    exit()    

frame_count = 0
MAX_FRAMES_TO_PORCESS = 150
DISPLAY_EVERY_N_FRAMES = 20

print(f"CamShift 추적 시작 (최대 {MAX_FRAMES_TO_PORCESS} 프레임, {DISPLAY_EVERY_N_FRAMES} 프레임마다 출력)...")

# 3. 첫 프레임에서 ROI 히스토그램 등록
ret, frame = cap.read()
if not ret:
    print("오류: 첫 프레임을 읽을 수 없습니다.")
    exit()

if w > 0 and h > 0: # 유효한 R이 설정된 경우에만 히스토그램 계산
    # 초기 추적 대상 영역(ROI) 추출 => hsv color 변경
    roi = frame[y: y + h, x: x+ w]
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 마스크 생성 : ROI 히스토그램 계산 시 노이즈 픽셀 제외(채도/명도 낮은 픽셀 제외)
    mask = cv2.inRange(roi_hsv, np.array((0.,50.,50.)), np.array((180.,250.,250.)))
    # roi_hsv의 최소색에서 최대색까지 범위를 검출하기 위한 마스크를 마드는 함수

    # ROI의 H(색상) 채널에 대한 히스토그램 계산 및 정규화
    # mask 사용, 히스토그램 계산(cv2.calcHist)
    roi_hist = cv2.calcHist([roi_hsv], [0], mask, [180], [0,180])
    # [0]:HUE(색), mask: 색검출 범위, [180]: bin 개수, [0,180]: h(HUE) 색검출 범위
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    # roi_hist(first): 원본배열, roi_hist(second) : 정규화된 배열
    # => 여기서는 원본 히스토그램 덮어 씌움
    # 0, 255: 히스토그램의 최소값 0 최대값 255
    print(f'초기 추적대상 영역 설정 완료: (x={x}, y={y}, w={w}, h={h})')
else:
  print("오류")
  exit()   

'''
노이즈 필터링 마스크
 - 마스크 역할
  * HSV 범위 필터링
  * 하한: (0., 50., 50.,): H는 모든 색상, S,V는 50 이상
  * 상한: (180., 255., 255.)
  * HSV (Hue, Saturation, Value: 명도(색의 밝기))
   # HUE(색상): 0-360도(openCV 0-179, 0-255 표현)
   # Saturation(채도) : 색의 순도, 0-1(0-255) 0에 가까울수록 어두움
   # V(밝기) : 0-1(0-255) 0에 가까울 수록 어두움
  * S와 V는 50 이상
   # S(채도)<50 : 회색(색이 거의 없다.)
   # V(명도)<50 : 너무 어둡다(검정에 가까움)
   # 이런 픽셀들 추적 잘 안됨(방해가 되는 noise)
'''

# 4. 비디오 처리 루프
while cap.isOpened() and frame_count < MAX_FRAMES_TO_PORCESS:
    ret, frame = cap.read()
    if not ret:
        break

    img_draw = frame.copy() # 원본 프레임 복사

    # 추적 진행
    if roi_hist is not None:
        # 전체 영상 BGR=>HSV 컬러 변환
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 마스크 생성: 채도(S)와 명도(V)가 낮은 픽셀을 제외하여 노이즈 제거
        target_mask = cv2.inRange(hsv, np.array((0.,50.,50.)), np.array((180., 255., 255.)))

        # Back Projection(역투영)
        # 전체 영상에 대해서 ROI 히스토그램 역투영
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)
        '''
        [hsv]: 입력 이미지, 히스토그램을 적용할 대상 (전체 비디오 프레임(이미지))
        [0]: 역투영에 사용할 채널 인덱스 => 색상만 사용
        roi_hist : 역투영 기준
        [0, 180] : hue의 값 범위
        1 : scale(추가 스케일링 안함)
        dst : 단일채널 (8bit) 이미지로 출력
        '''

        # 역투영 결과에 마스크 곱하여 노이즈 영역의 확률을 0으로 만들어 줌
        dst = dst * target_mask
        # mask는 범위 안의 색은 1(흰색) 범위 밖의 색은 0(검정)을 반환하기 때문에
        # 노이즈의 값은 0을 반환하기 때문에 역투영 결과에 곱해주면 0이 된다.

        # 마스크가 적용된 역투영 결과와 이전 추적 위치로 CamShift 추적 실행
        ret, track_window = cv2.CamShift(dst, track_window, termination)
        # ret : rotatedRact(회전된 사각형) 중심점(cx, cy) 크기 (w,h), 회전각 angle
        # track_window : 일반 bbox(bounding box) (x,y,w,h) => 다음 프레임의 입력으로 사용

        # 다음 프레임을 위한 track_window 업데이트
        x, y, w, h = track_window

        # 새로운 위치에 회전된 사각형 표시 (CamShift의 ret 사용)
        pts = cv2.boxPoints(ret)
        # 꼭지점 계산 ret의 회전 정보를 4개의 꼭지점 좌표로 변환
        # [[x1,y1], [x2,y12, [x3,y3], [x4,y4]]
        pts = np.int32(pts)
        cv2.polylines(img_draw, [pts], True, (0,255,0), 2)
        # True: 끝점과 시작점 연결
        '''
        계산(float) : 중심점 각도 등이 소수점으로 나옴
        좌표 변환(float) : 모서리 4개 좌표가 소수점으로 나옴
        형태 변환(int) : 소수점 버리고 정수로 변환
        그리기 : 픽셀 위에 그림
        '''

        result = np.hstack((img_draw, cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)))

    else:
      cv2.putText(img_draw, 'Tracking Failed. Check ROI', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1,
                  cv2.LINE_AA)
    
    if frame_count % DISPLAY_EVERY_N_FRAMES == 0:
        cv2.imwrite(f'{out_dir}\\pr0208_Camshift_{frame_count}.png', result)

    frame_count += 1

cap.release()
cv2.destroyAllWindows()