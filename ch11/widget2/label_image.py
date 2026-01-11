# button.py

from tkinter import *
from tkinter import PhotoImage

oroot = Tk()
oroot.geometry("400x300")

img1 = PhotoImage(file=r"D:\rokey\python\ch11\widget2\fruit.png")
img_label = Label(oroot, image=img1)
img_label.place(x=-20, y=-20)


obutton1= Button(oroot, text="PUSH1")
obutton2= Button(oroot, text="PUSH2")
obutton3= Button(oroot, text="PUSH3")
obutton1.pack()
obutton2.pack()
obutton3.pack()
oroot.mainloop()