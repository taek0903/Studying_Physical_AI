# geometry_grid.py

from tkinter import Tk
from tkinter import Button

oroot = Tk()
oroot.geometry("200x100")
obutton1 = Button(oroot, text="PUSH1")
obutton2 = Button(oroot, text="PUSH2")
obutton3 = Button(oroot, text="PUSH3")

obutton1.grid(row=1, column=0)
obutton2.grid(row=1, column=1)
obutton3.grid(row=0, column=4)


oroot.mainloop()