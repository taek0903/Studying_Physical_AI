# radio.py

from tkinter import Tk              # 부모 윈도우(위젯)
from tkinter import Button          # 버튼 위젯
from tkinter import Radiobutton     # 라디오버튼 위젯
from tkinter import IntVar          # 정수 변수

oroot = Tk()       # 부모 윈도우(위젯)
oroot.geometry("400x300")

radio_value = IntVar()          # 정수형 변수 생성
# radio_value.set(3)              # 정수값 설정 : 처음 선택되는 라디오 버튼
radio_value.set(-1)             # 아무것도 선택 안되도록 설정 방법

# value = radio_value.get()
# print(value)

lunch={0:"제육", 1:"돈까스", 2:"순대국밥", 3:"김치볶음밥"}

# radio_value => 어떤 버튼이 선택되어 있는지 저장할 변수
# variable => 클릭 된 버튼의 정보를 저장할 변수명 설정(선택 상태) 
# value => radio_value에 저장될 데이터 지정하는 인수(정수형 변수에 저장할 값)
orb1=Radiobutton(oroot,text=lunch[0],variable=radio_value,value=0)
orb1.pack()
orb2=Radiobutton(oroot,text=lunch[1],variable=radio_value,value=1)
orb2.pack()
orb3=Radiobutton(oroot,text=lunch[2],variable=radio_value,value=2)
orb3.pack()
orb4=Radiobutton(oroot,text=lunch[3],variable=radio_value,value=3)
orb4.pack()

def buy():
    value = radio_value.get()
    print(lunch[value])

obtn=Button(oroot, text="주문", command=buy)
obtn.pack()
oroot.mainloop()
