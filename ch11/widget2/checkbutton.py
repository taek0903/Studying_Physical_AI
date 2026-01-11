# checkbutton.py

from tkinter import Tk              # 부모 윈도우
from tkinter import Button          # 버튼 위젯
from tkinter import Checkbutton     # 체크버튼 위젯
from tkinter import BooleanVar      # 논리형 변수

oroot = Tk()
oroot.geometry("400x300")

check_value={}
coffee={0:"아메리카노", 1:"라떼", 2:"카푸치노", 3:"에스프레소"}

# check_value[0].set(False)
# value = check_value[0].get()
# print(value)

for i in range(len(coffee)):
    # variable => 체크박스 선택여부 저장
    check_value[i] = BooleanVar()     # 딕셔너리 새 값 할당
    ocheckbutton1 = Checkbutton(oroot, text=coffee[i], variable=check_value[i])
    # print("체크박스 선택 여부:", check_value[i].get())
    ocheckbutton1.pack(anchor="w")

def order():
    for i in range(len(coffee)):
        if check_value[i].get()==True:
            print(coffee[i])


obtn=Button(oroot, text="주문", command=order)
obtn.pack()

oroot.mainloop()