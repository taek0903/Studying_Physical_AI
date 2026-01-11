# opencv_module.py

import cv2

# 이미지 파일 읽기
image=cv2.imread(r"D:\rokey\python\ch20\opencv\rocket-8835386_640.png")

# 이미지 창에 표시
cv2.imshow('Load Image',image)
cv2.waitKey(0)          # 0: 무한대기
cv2.destroyAllWindows