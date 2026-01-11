# opencv_blur.py

import cv2
image=cv2.imread(r"D:\rokey\python\ch20\opencv\rocket-8835386_640.png")
print(image)

print('----------------------')
blurred = cv2.GaussianBlur(image,(15,15),0)
print(blurred)

# 블러 함수에서 커널 크기는 홀수
# 픽셀 주변 값 평균/가중평균 등으로 처리
# 홀수 크기 커널이여야 중앙 픽셀이 존재하여 대칭적 처리 가능

cv2.imshow('GaussianBlur', blurred)
cv2.waitKey(0)              # 0: 무한대긴, 대기 시간(ms)
cv2.destroyAllWindows       # 모든 OpenCY 창 닫기