# entry.py

from tkinter import Tk            # 부모 윈도우(위젯)
from tkinter import Entry         # 엔트리 위젯
from tkinter import Label         # 라벨 위젯
from tkinter import StringVar     # 문자열 변수

oroot=Tk()                        # 부모 변수(위젯)    
oroot.geometry("400x300")

# (문자열) 변수 값 위젯과 연결 가능
ostring=StringVar()
# textvariable : 값 변화 자동 반영
oentry=Entry(oroot, textvariable=ostring)
oentry.pack()

olabel=Label(oroot, textvariable=ostring)
olabel.pack()
oroot.mainloop()