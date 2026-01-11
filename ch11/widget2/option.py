# option.py

import tkinter as tk
root=tk.Tk()

option_list=['Option 1', 'Option 2', 'Option 3']

selected_option=tk.StringVar()

selected_option.set(option_list[0])

option_menu=tk.OptionMenu(root, selected_option, *option_list)
option_menu.pack()
root.mainloop()