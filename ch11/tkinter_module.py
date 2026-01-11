# tkinter_module.py

# import tkinter
# 1. 위젯 생성
# otk=tkinter.Tk()         # 부모 윈도우(위젯)
   # 버튼 위젯
# obtn=tkinter.Button(otk, text="click")
# 2. 위젯 배치
# obtn.pack()
# 3. 이벤트 바인딩
# otk.mainloop()

print('--------------------')
# from tkinter import *
# from tkinter import Tk
# from tkinter import Button
# otk=Tk()
# otk.geometry("100x150")
# otk.geometry("100x150+400+400")
# obtn=Button(otk, text="click")
# obtn.pack()
# otk.mainloop()

# Botton에 들어간 otk는 포함 관계
# 만약, 상속 개념으로 사용된다면,
# 다음과 같이 정의 되어야 함
# class Button(tkinter. Tk):
#   pass

print('--------------------')
from tkinter import Tk
from tkinter import Button

def hello(name):
    print(f"Hello {name}")

print('--------------------')
# 함수 호출 : 코드 실행
hello("Tom")

print('--------------------')
# 함수 객체 (괄호없이 사용 경우)
# 1. 변수에 저장(참조)
hello
f = hello
f("Jane")             # 함수 호출과 동일

# 2. 다른 함수에 전달
def call_func(func, value):
    func(value)
call_func(hello, "Alice")

# 3. 반환값으로 사용
def outer():
    return hello
g =  outer()
g("Kenneth")

print('--------------------')

def hi():
    print("Hi There")

otk=Tk()
otk.geometry("400x300")
# 3. 이벤트 바인딩
obtn=Button(otk, text="click", command=hi)
# obtn=Button(otk, text="click", command=lambda: hello("Python"))
obtn.pack()
otk.mainloop()

# 함수간 연산 가능(함수는 연상의 대상이 될수 있음)
