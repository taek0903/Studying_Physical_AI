# file_read.py

path = r"python\ch12\file1.txt"
# f=open(path, 'r',encoding="UTF-8")
f=open(path,encoding="UTF-8")

data=f.read()

print(data)

f.close()