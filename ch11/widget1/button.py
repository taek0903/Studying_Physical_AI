# button.py

from tkinter import *
oroot = Tk()
oroot.geometry("400x300")
obutton1= Button(oroot, text="PUSH1")
obutton2= Button(oroot, text="PUSH2")
obutton3= Button(oroot, text="PUSH3")
obutton1.pack()
obutton2.pack()
obutton3.pack()
oroot.mainloop()