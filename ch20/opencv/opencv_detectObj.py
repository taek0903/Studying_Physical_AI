# opencv_detectObj.py

import cv2

# 1. 이미지 로드
image = cv2.imread(r"D:\rokey\python\ch20\opencv\family.jpg")

# 2. 얼굴 검출을 위한 모델 확인
face_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier(face_path)
# 3. 이미지 그레이스케일로 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 4. 얼굴 검출 수행
faces = face_cascade.detectMultiScale(gray,
                                      scaleFactor=1.1,
                                      minNeighbors=5,
                                      minSize=(30,30))
print(faces)       # [x, y, w, h]
# 검출 얼굴 주위에 사각형 그리기
for (x, y, w, h) in faces:
    cv2.rectangle(image,
                  (x,y),
                  (x + w, y + h),
                  (255, 0, 0), 2)
    
# 이미지 창에 표시
cv2.imshow('Face Detection', image)
cv2.waitKey(0)              # 0: 무한대긴, 대기 시간(ms)
cv2.destroyAllWindows()       # 모든 OpenCV 창 닫기

# 파이썬 기본 문법 saeborn 오늘 자료는 중요하지 않다
# 파이썬에 대한 내용을 집중적으로 봐라