# file_readlines.py

path = r"python\ch12\file1.txt"
# f=open(path, 'r',encoding="UTF-8")
f=open(path, 'r',encoding="UTF-8")

lines=f.readlines()
for line in lines:
    print(line, end="")

f.close()