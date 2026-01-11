# file_path.py

# 절대경로(Absolute Path)
r"D:\rokey\python\ch12\file_path.py"

# 상대경로(Relative Path)
r"python\ch12\file_path.py"
r".python\ch12\file_path.py"

import os               # 내장모듈(운영체계)
print(os.getcwd())      # D:\rokey

# C:\
"..\rokey\python_class\ch12\file_path,py"

# 인코딩 : 텍스트(사람) => 바이너리(컴퓨터)
# 디코딩 : 바이너리(컴퓨터) => 텍스트(사람)

text1 = "A"       # 65
text2 = "a"       # 97
text3 = "가"      # ?

# 텍스트를 바이너리 형태로 변환 내장 함수
print(ord(text1))
print(ord(text2))
print(ord(text3))


# 바이너리를 텍스트 형태로 변환 내장 함수
print(chr(65))
print(chr(97))
print(chr(44032))