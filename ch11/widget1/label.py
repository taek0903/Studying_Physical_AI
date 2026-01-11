# label.py

from tkinter import *
oroot = Tk()
oroot.geometry("400x300")
olabel1=Label(oroot, text="적", bg="red", width=10)
olabel2=Label(oroot, text="녹", bg="green", width=20)
olabel3=Label(oroot, text="청", bg="blue", width=30)
olabel1.pack()
olabel2.pack()
olabel3.pack()
oroot.mainloop()