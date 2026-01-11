# opencv_color.py

import cv2

# 이미지 파일 읽기
image=cv2.imread(r"D:\rokey\python\ch20\opencv\rocket-8835386_640.png")

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 이미지 창에 표시
cv2.imshow('Grayscale Image', hsv)
cv2.waitKey(0)              # 0: 무한대긴, 대기 시간(ms)
cv2.destroyAllWindows       # 모든 OpenCY 창 닫기

# GRAY 방식
# 1채널만 사용(밝기만 표현) 255흰색 / 0 검정
# 이미지 연산의 양을 줄여서 속도를 높이는데 필요
# 변환 공식 : Gray=0.299 x R +0.587 x G + 0.114 x B
# 채널수 3개 => 1개 줄어듬

# HSV
# RGB와 마찬가지로 3개의 채널을 갖는 색상 이미지 표현법
# 3 채널 H(Hue, 색조), S(Saturation, 채도), V(Value, 명도)