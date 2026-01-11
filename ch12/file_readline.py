# file_readline.py

path = r"python\ch12\file1.txt"
# f=open(path, 'r',encoding="UTF-8")
f=open(path, 'r',encoding="UTF-8")

line1=f.readline()
line2=f.readline()
print(line1)
print(line2)

f.close()