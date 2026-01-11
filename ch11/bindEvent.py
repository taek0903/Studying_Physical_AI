# bindEvent.py

# from tkinter import Tk
# from tkinter import Button

# def order():
#     print("주문하세요.")

# oroot = Tk()
# oroot.geometry("400x300")
# # 버튼 위젯
# obutton1=Button(oroot, text="주문", command=order)
# obutton1.pack()
# oroot.mainloop()

# print("----------------------------")
# # 이벤트 바인드 방식2 : bind( 함수 사용)
from tkinter import Tk
from tkinter import Button

def order(event):
    print("주문하세요.")

oroot = Tk()
oroot.geometry("400x300")
# 버튼 위젯
obutton1=Button(oroot, text="주문")
obutton1.pack()

# 객체명.bind("약속된 문자열", 함수객체)
obutton1.bind("<Button-1>", order)

# oroot.mainloop()

# Tkinter 주요 이벤트 종류
# 이벤트                설명
# --------------------------------------------
# <button-1>            마우스 왼쪽 버튼 출력
# <Double-Button-1>     마우스 더블 클릭
# <KeyPress-a>          특정 키 입력 : 'a'

# print("----------------------------")
# from tkinter import Tk
# from tkinter import Button

# def order():
#     print("주문하세요.")
# root=Tk()
# btn=Button(root, text="주문", command=order)
# btn.pack()
# root.mainloop()

from tkinter import Tk
from tkinter import Button
from tkinter import Label
def msg():
    print("start tkinter")
root = Tk()
root.geometry("400x300")
mlabel=Label(root, text="시작레이블")
mlabel.pack(side="top")
obutton1=Button(root, text="시작버튼", command=msg)
obutton1.pack(side="bottom")

# 객체명.bind("약속된 문자열", 함수객체)
obutton1.bind("<Button-1>", order)