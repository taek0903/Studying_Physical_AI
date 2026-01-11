# opencv_detectEdge.py

import cv2

# 이미지 파일 읽기
image=cv2.imread(r"D:\rokey\python\ch20\opencv\rocket-8835386_640.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, threshold1=70, threshold2=150)

cv2.imshow('detectEdge', edges)
cv2.waitKey(0)              # 0: 무한대긴, 대기 시간(ms)
cv2.destroyAllWindows       # 모든 OpenCY 창 닫기